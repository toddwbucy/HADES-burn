# Structural-Embedding Modeling Lessons (GraphSAGE / RGCN)

> **HADES methodology corpus.** Empirical findings from training `hetero_sage`
> over a code ↔ doc **conformance knowledge graph** (~5,500 nodes, ~6,000 edges)
> during the WeaverTools build, 2026-06-09. Graph-modeling and methodology lessons
> for HADES structural-embedding training, generalizable beyond the specific build.
>
> **This corrects a claim in [`the-method-hades.md`](the-method-hades.md):** the
> directory backbone is a *traceability / connectivity* overlay, **not** a training
> signal. Injecting the directory tree as training edges causes hub oversmoothing
> (Mode 1). Exclude containment from `relation_order`; encode hierarchy as node
> **features**. Connectivity-for-querying and structure-for-learning are opposite
> requirements.

## TL;DR

The `graph-embed` pipeline is mechanically sound (GPU train → export → neighbors),
but the quality of the structural embeddings is governed entirely by the graph's
**relational design**. Five distinct modeling errors, **each independently
sufficient to cause total embedding collapse** (every distinct node → the same
vector, cosine ≈ 1.0). Aggregate AUC hides this; diagnose with **per-node-type
embedding diversity**.

Evidence trail (same graph, same features; only `relation_order` / modeling changed):

| Configuration | test AUC | code-file embeddings | doc-side embeddings |
|---|---|---|---|
| all edges, no non-code features | 0.19 | collapsed (cos 1.0) | collapsed |
| + featurized all node types | 0.21 | collapsed | collapsed |
| − symbol layer (defines/imports) | **0.66** | collapsed | **diverse** (cos 0.26–0.82) |
| − directory/crate hubs too | 0.63 | collapsed | diverse |
| + reversed embodiment / inflow | **0.76** | collapsed | **diverse (0.39)** |

The doc/conformance half learned cleanly; the code half never did. The five causes
below explain it: modes 1-4 are graph-modeling, mode 5 is a suspected
model-internal defect, softened to **pending** by a second evidence trail
(2026-06-10, candle DB) that reproduced an identical collapse and recovered it
with a graph-level fix alone. See "Second evidence trail" below.

## The five collapse modes

### 1. Hierarchy / position modeled as EDGES → hub oversmoothing

- **Symptom:** all nodes funneling through a shared ancestor collapse to one vector.
- **Cause:** the filesystem directory tree was injected as `contains` edges. Every
  file funnels through shared ancestor directories up to a single root — a hub/star
  topology. Message-passing through hubs averages neighbours until siblings are
  identical (textbook oversmoothing).
- **Fix:** model hierarchy/position (directory path, module, package) as **node
  features / metadata**, not edges. Keep containment edges only as a **traceability
  overlay** for querying ("trace node to root") and **exclude them from
  `relation_order`** (the training scope).
- **Principle:** *edges define what a node aggregates from; features define what a
  node is.* Position data belongs in features. **Connectivity-for-querying and
  structure-for-learning are opposite requirements** — a single-rooted tree is
  ideal for "trace to source" and toxic for learning. Separate them.

### 2. A DOMINANT relation type → objective capture

- **Symptom:** the entire graph collapses; AUC below random (~0.2).
- **Cause:** one relation type (`defines`/`imports`, file→symbol) was ~92% of all
  edges (5,530 of 6,000). The link-prediction objective is captured by the
  dominant, dense, low-signal relation, and the model satisfies it by collapsing
  everything.
- **Fix:** balance `relation_order`, or exclude / down-weight dominant low-signal
  relations. Removing the symbol layer here **tripled** AUC (0.19 → 0.66) and let
  the conformance side learn.
- **Principle:** the GNN optimises whatever relation dominates the edge count.
  Curate `relation_order` for **signal balance**, not completeness.

### 3. SINK node types (all-outgoing edges) → no inflow to aggregate

- **Symptom:** a node type collapses to a constant **even with diverse input
  features and diverse neighbours**.
- **Cause:** GNN nodes learn by aggregating **incoming** neighbours. If a node
  type's edges all point outward (here: code `embodies → spec`, `belongs_to →
  crate`, `imports → symbol`), it has no inflow → nothing to aggregate → it
  collapses to the node-type's bias vector. The directory `contains` edges
  (dir → file) were the files' *only* inflow; removing them (fix #1) exposed the
  sink.
- **Fix:** ensure every trained node type has **incoming** edges from diverse
  nodes. Make semantic edges bidirectional, or add the reverse relation (e.g.
  `spec realized-by → file`), so terminal/leaf nodes *receive* signal.
- **Principle:** model the graph so signal **flows into** every node type you want
  embedded, not just out of it. Audit edge directionality per node type. (The doc
  side learned precisely because it is richly bidirectional: assertions receive
  from files, axioms from assertions, documents contain assertions.)
- **Refinement (2026-06-10, candle reproduction): self-referential inflow is no
  inflow.** A reverse relation only helps when the reverse source's features
  differ from the target's. In the candle v1 run, files had 10,439 incoming
  `defined_in` edges (reverse of defines) and collapsed anyway, because a
  symbol's loader feature IS its parent file's pooled vector: every message a
  file received was a copy of its own feature. One dense **cross-file** relation
  (`imported_by`, symbol -> file, 4,549 edges) recovered both node types
  completely. The doctrine wording is therefore: ensure every trained node type
  has incoming edges **whose sources carry information the target does not
  already have**.

### 4. FEATURELESS node types → silently skipped

- **Symptom:** non-code node types get no embedding; loader logs `embeddings
  skipped (null or wrong dimension)`.
- **Cause:** the loader reads `d.embedding` (feature_dim, e.g. 2048) per
  **non-code** node and skips null/wrong-dim. Code nodes are mean-pooled from their
  chunk embeddings (`codebase_embeddings` by `file_key`); **every other node type
  must have the `embedding` field set explicitly.**
- **Fix:** featurize every participating node type — text-embed text-bearing nodes
  (documents, assertions, axioms, smells, concepts), mean-pool container nodes
  (directories) from their descendants. Verify **0 "embeddings skipped"** before
  trusting a run.
- **Principle:** a featureless node type contributes nothing; check feature
  coverage per type.

### 5. A node type with a SEPARATE feature-loading path can collapse alone (HADES model-internal)

> **Status (2026-06-10): suspected, not confirmed. Pending a dense-cross-flow
> re-test.** The candle reproduction (second evidence trail below) recovered an
> identical codebase_files collapse with a graph-level fix alone, through the
> same `load_codebase_node_features` path that was suspect here. Re-reading this
> trail with that lens: the "+ reversed embodiment" run gave code files
> cross-TYPE inflow at roughly 10^2 edges against ~5.5k total, two orders of
> magnitude sparser than the 4.5k cross-file edges that worked on candle. The
> graph-level explanation (informationally degenerate inflow, mode 3 refinement)
> may suffice. Before concluding the codebase feature path is defective, re-test
> this graph with a dense cross-file relation (an imported_by equivalent). The
> original evidence is kept below, unchanged.

- **Symptom:** after fixing modes 1–4, one node type still collapses (cosine ≈ 1.0)
  while every other type embeds diversely — and it stays collapsed under *every*
  structural change.
- **Evidence (WeaverTools build):** `codebase_files` collapsed across all of:
  symbol layer in/out, directory hubs in/out, and **reversed embodiment added** (so
  the type has diverse inflow). Meanwhile its **input features are verified
  diverse** — both the raw `codebase_embeddings` chunks (cos 0.45–0.88) and the
  mean-pooled per-file vectors (cos 0.50–0.94). The doc-side types embed with high
  diversity. AUC climbed 0.19 → 0.66 → 0.76 from graph fixes alone, confirming the
  *graph* is correct.
- **The discriminator:** `codebase_files`/`codebase_symbols` are loaded by a
  **different code path** — `load_codebase_node_features` (mean-pool of chunk
  embeddings) — whereas all other types read the node's `d.embedding` field via
  `load_collection_embeddings`. Identical diverse 2048-dim inputs; only the path
  differs; only the mean-pool-path type collapses.
- **Lead for HADES:** suspect `load_codebase_node_features` (or the per-node-type
  projection/bias for the codebase node type in `hetero_sage`). A diagnostic
  workaround to confirm: materialize the mean-pooled vectors as a literal
  `embedding` field on `codebase_files` and route them through the same
  `load_collection_embeddings` path as other types; if they then diverge, the
  defect is in the codebase feature path, not the model core.
- **Principle:** when collapse is isolated to one node type that is verifiably
  well-featured and well-connected, the cause is type-specific *plumbing* (its
  feature path / type embedding), not the graph. Diagnose by **routing its features
  through the common path** and comparing.

## Second evidence trail: candle DB reproduction and recovery (2026-06-10)

A fresh `hetero_sage` run over the `candle` database (huggingface/candle fork,
806 files / 10,439 symbols / ~26k edges, Jina-v4 2048-dim features) reproduced
total collapse and then **fully recovered it with a single graph-level
variable**. This is the cleaner of the two trails: one variable apart, same
features, same checkpoint dir, same GPU.

| | v1 | v2 |
|---|---|---|
| relation_order | defines, defined_in (reverse), imports, calls | + **imported_by** (reverse of imports, symbol -> file, 4,549 edges) |
| test AUC | 0.18 (below random) | **0.74** |
| codebase_files diversity (mean pairwise cos, n=40) | **1.0000, total collapse** | **0.241, diverse** |
| codebase_symbols diversity | 0.947, near-collapse | **0.015, highly diverse** |

Sanity check: post-v2, `graph-embed neighbors` on `gpt_oss.rs` returns other
candle model files (mobileclip, mmdit blocks, gemma4_vision, mimi_seanet),
which is the correct structural cluster, since candle model files are
deliberately cookie-cutter. Within-cluster neighbor similarities saturate at
1.0, so ties remain inside a cluster.

What this trail establishes:

1. **Mode 3 refinement confirmed.** v1 was not a naive sink: files had 10,439
   incoming `defined_in` edges and collapsed anyway, because that inflow was
   self-referential (see the refinement bullet under mode 3). One dense
   cross-file relation recovered everything.
2. **Mode 5 softened.** No model-internal defect is needed to explain this
   collapse, and the recovery ran through the suspect loader path. See the
   status note on mode 5.
3. **Input-feature degeneracy is real, as an enhancement target rather than a
   defect.** `load_codebase_node_features` gives every symbol its parent file's
   pooled vector, so an 11k-node graph had only ~642 distinct input features.
   v2 shows the GNN can differentiate via structure anyway, but the degeneracy
   is a handicap and plausibly contributes to within-cluster saturation. Two
   loader enhancement candidates: **per-symbol pooling** (pool a symbol's own
   overlapping chunks via `codebase_chunks.symbols[]` instead of the whole
   file) and **`d.embedding` precedence in the codebase branch** (prefer a
   literal embedding field when present, fall back to pooling). The second also
   unlocks the diagnostic workaround mode 5 prescribes, which is currently
   impossible because the loader dispatch is unconditional by collection name.

Repro: database `candle` (schema in `~/git/candle-hades/schema.yaml`, reverse
edges materialized from the forward collections with `rev_` keys). v1 = drop
`codebase_imported_by_edges` from `relation_order`, `schema apply --force`,
retrain. v2 = restore it.

## Diagnostic methodology

- **Do not trust aggregate AUC alone.** It hid that the doc side learned while the
  code side collapsed (the AUC was carried by the half that worked).
- **Measure per-node-type embedding diversity** on the **exported** embeddings:
  mean/min pairwise cosine within each node-type. Collapse ≈ cosine 1.0. This
  localises the failure to specific types.
- **Isolate one variable at a time** (features → dominant relations → hubs →
  directionality). Each rules a distinct cause in or out. All five above were found
  this way.

## Operational gotchas

- **Schema integrity:** hand-editing the live `hades_schema` (relation_order,
  num_relations, named-graph edge definitions) breaks the trainer's
  `schema_checksum` and the `num_relations == len(relation_order)` invariant.
  Always change the schema via `schema apply --force` (recomputes the checksum),
  never by direct document edits.
- **Named graphs vs training scope:** an edge collection can live in only one named
  graph (ArangoDB err 1921). The **training scope is `relation_order`**,
  independent of named-graph membership (traversal scope). `schema apply` skips
  reconciling existing named graphs ("skipped_existing") — edge definitions on a
  pre-existing graph must be updated out of band.
- **checkpoint-dir:** must be writable by **both** the caller and the `hades`
  training-service user (it runs as a separate uid and both reads/writes there).
- **Featurizing at scale:** start the embed service (`embed service start`) before
  batch-embedding hundreds of nodes, so the model stays resident.

## Recommended methodology / tooling additions

1. **Schema lint (pre-train):** warn if any single relation exceeds ~N% of edges
   (imbalance, mode 2). Warn if a trained node type has zero incoming edges
   (sink, mode 3), and also if a trained node type's only inflow sources **share
   its feature provenance** (the `defined_in` trap from the candle trail: the
   mode-3 refinement is statically detectable, since the loader's
   pooled-from-parent provenance is known per collection). Warn if a node type
   has <100% feature coverage (mode 4), or if a pure hierarchy/containment
   relation is included in `relation_order` (hub risk, mode 1).
2. **Post-train report:** per-node-type embedding-diversity (collapse detector),
   alongside AUC.
3. **Modeling doctrine:** hierarchy/position belong in features, not edges.
   Containment/membership edges are a traceability overlay, excluded from
   `relation_order`. Semantic edges run bidirectional so leaf/code node types
   receive signal, with the qualifier from the mode-3 refinement: the inflow
   sources must carry information the target does not already have.
4. **Loader enhancements (from the candle trail):** per-symbol pooling, and
   `d.embedding` precedence in the codebase branch with pooling as fallback
   (also unlocks the mode-5 diagnostic).
5. **Pending re-test:** the WeaverTools graph with a dense cross-file relation
   (an imported_by equivalent), to settle mode 5.

## See also

- [`the-method-hades.md`](the-method-hades.md) — the HADES-explicit method; its
  *Structural embeddings* section carries the short-form doctrine and its
  *structural backbone* section is corrected by Mode 1 here.
- [`../graph-methodology.md`](../graph-methodology.md) — `relation_order`,
  `feature_dim`, `model_type` (Layer 3).
