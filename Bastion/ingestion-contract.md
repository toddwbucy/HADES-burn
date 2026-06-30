# HADES Ingestion Contract

Version 0.1 (draft), 2026-06-09. Tool-level document: this contract describes
HADES the tool. It is method-neutral. Any methodology (the bastion method or
another) is a client discipline layered on top of this surface.

Items tagged **(new)** are contract obligations the current pipeline does not
yet meet. They are the ingestion backlog, stated here so the gap is visible
rather than implied.

---

## 1. Scope: document processing, not document production

HADES is a document-processing component inside a larger context-management
system. The boundary is the handoff: a valid document in a supported format
goes in, a processed graph comes out.

HADES owns everything downstream of the handoff: ingestion pipelines (code,
PDF, general documents), chunking, the agent-assisted dissection that proposes
bridges and assigns cells during ingest, storage, retrieval, constraint
enforcement, and the gate vocabulary.

Everything upstream of the handoff is not a HADES concern. Session harvesting,
decision-record authoring, and all other document production live in the
client system (WeaverTools, a human editor, anything else). HADES is
provenance-blind at the input: it never knows or cares whether an agent or a
person wrote the document.

One consequence stated plainly: **HADES does not passively accumulate
rationale.** The graph holds the why only if someone wrote it down and handed
it across this boundary. A methodology that claims why-capture must supply the
documents.

---

## 2. Supported inputs and what each triggers

| Input | Pipeline | Output |
|---|---|---|
| Rust / Python source tree | `codebase ingest`: AST dissection (syn + rust-analyzer for Rust, Python AST for Python) | `codebase_files`, `codebase_chunks`, `codebase_embeddings`, `codebase_symbols`, plus `defines` / `imports` / `calls` edges |
| Unparsed-extension files | `codebase ingest --unparsed-ext`: size-chunk fallback | chunks + embeddings, no symbol extraction |
| Markdown / plain text | `ingest`: chunk + embed | document-profile collections |
| PDF | `extract` (extractor service) then `ingest` | document-profile collections |
| Decision record | the document pipeline, unmodified | a document node with `kind: decision_record` |

Notes:

- **Ignore rules are honored.** `codebase ingest` respects `.gitignore` and
  related files, which means ignored files are invisible to the graph by
  design. Callers who want them must say so.
- **The decision record is a document kind, not a special pipeline.** It is a
  markdown document with structured front matter (decision, status, why,
  alternatives considered) and a `kind` field. It passes the same gate as
  every other artifact. Nothing about its provenance is recorded or consulted.
- **Agent-assisted dissection** (bridge proposal, cell assignment) is a
  pipeline stage that may be executed by an attached agent driving HADES
  commands. Its outputs are bound by this contract regardless of executor:
  every proposed edge lands with `provenance: asserted` (section 3).
- **(new) Directory backbone.** `codebase ingest` emits `codebase_directories`
  nodes and containment edges derived from `rel_path`, excluded from
  `relation_order` by default. Until this lands, callers materialize the
  backbone themselves.

---

## 3. Confidence metadata per extracted element

The reason this section exists, in one sentence: **absence of an edge must be
distinguishable from a missed extraction, and that distinction can only be
recorded at ingest time, because at query time the information is gone.**

The known limitation "an absence query cannot tell a real orphan from a missed
extraction" is hereby reframed: it is an ingestion defect that the query layer
inherits. The fix lives here or nowhere.

Per element:

- **Symbols** carry `extraction.source`: which parser produced them (syn first
  pass, rust-analyzer enrichment, fallback chunker). This already exists as
  the `source` field.
- **(new)** `extraction.confidence` where the extractor can score itself, on
  any element type whose extraction is heuristic rather than exact.
- **(new) Per-file coverage summary:** declarations seen versus symbols
  extracted, written on the file node, so a file with low coverage flags its
  own absences as suspect-extraction rather than real orphans.
- **Bridge edges** carry three fields: `evidence` (the concrete basis for the
  bridge), `confidence` (the proposer's own estimate), and `provenance`, an
  enum with exactly three values:

  `asserted` | `spot_checked` | `human_confirmed`

  Default at ingest is `asserted`. Audit review upgrades to `spot_checked`.
  Ratification upgrades to `human_confirmed`. No fourth value, no skipping
  states downward except through the exit gate.

---

## 4. What the caller is promised

1. Every non-ignored file under the ingest root is visited.
2. Stable keys: `_key` = `rel_path` with `.` and `/` mapped to `_`. Same root,
   same keys, across re-ingests.
3. Idempotent re-ingest: changed files are purged and rebuilt
   authoritatively, unchanged files are skipped.
4. The code pass is distinct and explicit. No run reports completeness over
   code it was not pointed at.
5. **(new)** Zero structural orphans: once the backbone lands, every code node
   hangs off the directory tree by construction.

---

## 5. What the caller is not promised

1. **Extraction recall of 1.0.** A missing edge can be a missed extraction.
   The confidence and coverage metadata of section 3 exist so the query layer
   can separate "no edge, high coverage" (a real orphan) from "no edge, low
   coverage" (suspect extraction).
2. **Bridge completeness or correctness.** Bridges emitted at ingest are
   proposals (`provenance: asserted`) until audited or confirmed.
3. Cross-document semantic deduplication.
4. Rationale capture (section 1). The why arrives as a document or not at all.

---

## 6. Verdict provenance mix

Any verdict computed over bridge edges must report the provenance mix of the
edges it rests on: counts of `asserted`, `spot_checked`, `human_confirmed`. A
verdict built on unaudited assertions is visibly different from one built on
confirmed structure. The gate verb group **(new)** consumes this requirement:
`gate audit --sample N` pulls a load-bearing-weighted sample of bridge edges
for human review and writes the upgrades.

---

## 7. Canon-node metadata

Ratification stamps on canon nodes (axioms, smells, decision records) use a
set, not a scalar: `ratified_by` is an array of identities, alongside a
`ratified` date. Quorum and dispute semantics are deliberately undefined in
this contract version. The set form exists so a future multi-adjudicator
method is a schema extension instead of a schema migration.

---

## 8. Versioning

This contract is the coupling surface between HADES and any client system.
Changes to it are breaking changes and get a version bump. The **(new)** tags
above constitute the v0.1 ingestion backlog: backbone emission, extraction
confidence, per-file coverage, provenance field defaults, gate audit.
