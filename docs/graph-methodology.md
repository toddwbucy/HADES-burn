# Graph Methodology — IS / IS-NOT Axiom-Gated Knowledge Graphs

> The canonical, reusable method for constructing a HADES knowledge graph from a
> foundation document and a codebase. This is the **context-engineering
> methodology** behind graphs like NestedLearning (`NL`) — the schema and edge
> structure are *the engineering*, not an afterthought.

The graph is built to behave as an **immune system**: a concept, a smell, or a
file *earns* its place only by tracing — through ratified edges — back to an
axiomatic identity. What cannot trace is not "untagged"; it is **suspect**, and
the graph makes that absence queryable. The method is invariant across the
document it is pointed at: the *content* changes per source, the *structure*
does not.

---

## Core invariants (the spine)

1. **Document-first, code-last.** The foundation document is dissected into typed
   concept nodes *before* any code is ingested. Code joins **last** and is held
   to the same identity test the concepts were.
2. **Membership is earned by connection, never asserted.** Every concept node
   carries **both** poles of the identity gate: a `basis` edge to the `IS`
   container **and** a `validated-against` edge to the `IS_NOT` container. Two
   edges per concept — near-parity in their counts is the health signature.
   (These edges are the *evidence* an artifact proves with; the artifact, not
   the individual assertion, is the unit of ratification — see [The ratification
   rule](#the-ratification-rule--the-unit-of-proof-is-the-artifact).)
3. **Provenance is a hard gate on axioms.** A principle enters an axiom container
   only with a **verbatim source quote** + structured provenance **and human
   ratification**. No ratified quote → no entry.
4. **Smells are derived from axioms, not invented.** Enforceable code rules trace
   to the specific `IS`/`IS_NOT` principles they enforce or guard.
5. **Non-connection is the signal.** The payoff is two queries over *absence*:
   code with no bridge edge (embodies nothing documented) and concepts with no
   structural embodiment (a claim with no implementation).

---

## The ratification rule — the unit of proof is the artifact

A node earns its place one of two ways, and which one depends on a single
question: **does it exist in the world on its own?**

**Artifacts prove.** A node that corresponds to a real-world artifact — a file,
a git issue, a spec, a code unit — must prove against the axioms to be
**ratified**, and it is judged *as a whole*:

- A **document** proves *holistically*: its substantive claims bridge to the
  gate (`basis` + `validated-against`) **and** all of its configuration and
  implementation details are accounted for, with none violating.
- **Code** proves via smell-compliance plus intact provenance to an
  already-ratified spec.

**Details inherit.** A node born *from* an already-ratified artifact — its
extracted claims, its configuration, its implementation details, none of which
are independent artifacts — **inherits** that artifact's verdict. It is not
gated again on its own.

So the unit of proof is the artifact. The provenance chain is a chain of
artifacts — e.g. `Issue → PRD → Spec → code` — each **independently proving**
and each linked to its parent for lineage. Everything extracted from an artifact
rides that artifact's verdict.

**The criterion is existence, not hierarchy.** Anything with its own footprint
in the world — a file on disk, an issue in the tracker, a commit — has to stand
on its own against the axioms, because it can be changed *independently of its
parent* and can therefore *independently drift*. Anything that exists only as a
description inside a ratified artifact cannot drift on its own, so it rides the
parent's verdict.

**Lineage and ratification are different edges.** A spec links to its PRD for
*provenance* (a lineage edge — an edge collection with
`materialize_strategy: lineage`, parent → child), but it earns its *own*
ratification — the spec is a separate file someone can edit without touching the
PRD. Never conflate the lineage edge with the gate edges (`basis` /
`validated-against`, artifact → axiom): one records where a node came from, the
other records that it proved.

This sharpens two things in the layers below:

- **The compliance unit is the artifact, judged whole.** The extracted
  assertion-nodes that bridge to the gate are the *evidence* for the artifact's
  verdict — not a swarm of independently-ratified per-assertion units.
- **Suspect has a precise definition** (Layer 5): an artifact that fails to
  prove as a whole; a detail node whose parent artifact isn't ratified or whose
  provenance chain is broken; or any node that violates an axiom. **Indirection
  alone is never suspect** — a detail sitting two hops from the gate via its
  ratified parent is normal inheritance, not a defect.

> A node that *is* an artifact but whose parent isn't yet ratified is **pending
> parent ratification**, not suspect — distinct states. Calibrating the gate on
> a sample of a document's assertions proves the *mechanism*; it does not ratify
> the document, which requires accounting for the whole file.

---

## The five layers

### Layer 1 — Foundation document, dissected relationally

Per source, extract a family of typed **concept collections** — definitions,
equations, abstractions, axioms, lineage — one node per extracted concept. Wire
them with:

- **Internal relational edges** (within a source): e.g. `*_equation_depends_edges`
  (equation → equation), `*_definition_source_edges` (definition → equation),
  `*_signature_equation_edges` (code signature → equation).
- **Cross-source edges**: `*_cross_paper_edges`, `*_lineage_chain_edges`,
  `*_reframing_link_edges` (e.g. external-framework concept → this framework's).

The output is a relational map of *what the document says*, before any judgement
of identity.

### Layer 2 — The IS / IS-NOT axiom gate (the identity layer)

A single collection (e.g. `nl_axioms`) holds **exactly two** `axiomatic_container`
documents — `<NS>_IS` and `<NS>_IS_NOT` — each carrying a `principles[]` array.

**Every principle is grounded** in a `verbatim_source`:

```yaml
verbatim_source:
  source_id:       "<arxiv id or document id>"
  quote:           "<exact verbatim text>"
  location:        "<section / page / line>"
  authority_basis: human_approved      # the only accepted basis
  ratified:        "<YYYY-MM-DD> by <project lead>"
  reading:         direct               # not inferred
```

Nothing enters the gate without a ratified verbatim quote. The four edge types
that wire concepts to the gate:

| Edge | Direction | Meaning |
|------|-----------|---------|
| `*_axiom_basis_edges` | concept → `IS` | what this concept **is** (`source_field: axiom_basis`) |
| `*_validated_against_edges` | concept → `IS_NOT` | what this concept is tested **against** |
| `*_structural_embodiment_edges` | `IS` → concrete definition | the `IS` pole made concrete |
| `*_axiom_inherits_edges` | per-source container → framework `IS`/`IS_NOT` | the per-source container **derives from** the framework identity (arrow points to what it inherits) |

A concept missing **either** of its two gate edges is a defect, not a style
choice. (In `NL`: 607 `basis` + 606 `validated-against` — the near-parity is the
tell that membership was genuinely earned on both poles.)

### Layer 3 — Smells inferred from the axioms

A smell collection (e.g. `nl_code_smells`) turns axioms into **enforceable
rules**. Each smell record carries:

- `axiom_basis` — which `IS` container it derives from.
- `is_axioms` / `is_not_axioms` — the principle ids it **enforces** / **guards**.
- `validated_against` — the `IS_NOT` linkage.
- `verbatim_basis` — the grounding quote, or the reference pseudocode (e.g. the
  algorithm listing) the axiom traces back to.
- `forbidden_patterns[]` — concrete code signatures that violate it.
- `scope` — `python` | `rust` | `cuda` | …
- `origin` — `operational` (promoted from a recurring real bug) vs `derived`
  (read straight from an axiom).

A smell is promoted to enforcement when it earns it — e.g. a smell tracking
unbounded gradient accumulation is promoted from `derived` to `operational`
after the same bug recurs three times in 48 hours.

### Layer 4 — Code ingested and bridged to the concept graph

Ingest the codebase the standard way (`hades codebase ingest` → `codebase_*`
collections + structural edges; see [codebase-graph-ontology.md](codebase-graph-ontology.md)).
Then add **bridge edges** from code to the concept graph:

| Bridge edge | Direction | `rel` | Evidence |
|-------------|-----------|-------|----------|
| `*_code_spec_edges` | file → spec | `implements` | `confidence` + `evidence` (e.g. filename stem == spec key) |
| `*_code_equation_edges` | file → equation | `cites` | extracted from docstrings, with line numbers |
| `*_smell_compliance_edges` | file → smell | `complies` \| `violates` | enforcement basis |

Code now inherits the same identity test: a file with no bridge edge to a spec,
equation, or axiom embodies nothing documented.

### Layer 5 — The suspect set

A named graph (e.g. `nl_code_paper`) unifies the code edges with the bridge
edges. The **suspect set** (per [The ratification
rule](#the-ratification-rule--the-unit-of-proof-is-the-artifact)) is every node
that is an artifact failing to prove as a whole, a detail node whose parent
artifact isn't ratified or whose chain is broken, or any node that violates an
axiom. Indirection alone is never suspect. It surfaces as **queries over
absence**:

- **Orphan code** — a `codebase_file` with **no** bridge edge: it embodies
  nothing documented, or should not exist.
- **Unembodied concept** — a concept with **no** `structural_embodiment` edge: a
  claim the codebase has not yet implemented.
- **Broken provenance** — a detail node whose parent artifact is unratified or
  whose lineage chain back to a ratified artifact is severed.

The "living divergence register" stops being a hand-maintained list and becomes a
standing query. Non-connection is the signal.

---

## `relation_order` — what trains

For structural embeddings (see [declarative-schema.md](declarative-schema.md)),
`relation_order` scopes which relations the model trains on. Include the
**semantic + structural** relations (concept edges, gate edges, bridge edges, and
the `codebase_*` edges). **Exclude** process / project-management relations — a
training graph carries *what the work means*, never *who/when/status* (the
human/agent-UI boundary). `feature_dim` is the node feature width (2048 for Jina V4).
`model_type` selects the encoder: `rgcn` (transductive) or `hetero_sage`
(inductive — embeds nodes added after training without a retrain; use it for a
continuously-growing graph).

---

## Procedure for a new graph

1. **Provision a HADES-owned database** (never a production research DB):
   `hades --db _system db create-database <DB>`.
2. **Author the foundation layer** — dissect the document into concept
   collections + internal/cross-source edges (Layer 1).
3. **Build the gate** — author the `IS` / `IS_NOT` containers with
   **ratified verbatim** principles; wire every concept with its `basis` +
   `validated-against` edges (Layer 2). *Ratification is a human step.*
4. **Derive the smells** from the axioms (Layer 3).
5. **`schema apply`** a YAML declaring the collections, edge definitions, the
   named graph, `relation_order`, `feature_dim`, and `model_type`
   (see [declarative-schema.md](declarative-schema.md)).
6. **Ingest the codebase last** and add the bridge edges (Layer 4).
7. **Run the suspect-set queries** (Layer 5); optionally train structural
   embeddings (`graph-embed train`; `model_type` picks RGCN or inductive
   GraphSAGE) once every trained node carries a feature.

> Per-graph *design* (the specific concept families, principles, smells for one
> document) belongs in that graph's **companion document**, not in code or this
> spec. This file is the invariant method; the companion document is one
> application of it.

---

## Canonical instance

`NestedLearning` (`NL`) is the reference implementation: per-source concept
families (`hope_*`, `atlas_*`, `titans_*`, `nl_*`), the `NL_IS` / `NL_IS_NOT`
gate (607 basis / 606 validated-against / 13 structural-embodiment / 16 inherits),
the `nl_code_smells` layer, the `nl_code_{spec,equation}` + `nl_smell_compliance`
bridges (89 / 151 / 12), and the `nl_code_paper` named graph.

---

## See also

- [declarative-schema.md](declarative-schema.md) — the `schema apply` YAML format (`relation_order`, `feature_dim`).
- [codebase-graph-ontology.md](codebase-graph-ontology.md) — the `codebase_*` collections + structural edges.
- HADES CLI skill (`~/.claude/skills/hades/`) — the command mechanics this method drives.
