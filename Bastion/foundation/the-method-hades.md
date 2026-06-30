# The Method — HADES

The HADES-wired version of **The Method**. Same workflow; every step is a concrete
`hades` command and the actual collection vocabulary, so it runs reliably on a
local project without guessing. The backend-agnostic write-up is the genus; this
is the species you run. See also [`bastion-playbook.md`](bastion-playbook.md)
(session flow + division of labor) and [`../graph-methodology.md`](../graph-methodology.md)
(the full Layer-3 ontology).

> Run it where the externality test passes — infrastructure meant to outlive its
> makers. For throwaway code, skip it; the method is not free.

---

## HADES conventions (always)

- `hades --db <db> <cmd>` — there is no implicit default; set `--db` every time.
- JSON → stdout, logs/progress → stderr. Append `2>/dev/null` and pipe to `jq`.
- **One writable target:** your project's method DB — `bident_burn` for HADES-Burn
  itself, `<project>_burn` for another. Production research DBs are ACL read-only;
  that's the backstop, not a convention you observe.
- `db aql` is **read-only** (mutating AQL is rejected). Writes go through the
  structured `db` ops. The agent never runs a ratifying `db insert` on its own
  (see The human gate).

---

## The graph shape

`codebase ingest` builds the **code half** automatically: `codebase_files`,
`codebase_chunks`, `codebase_embeddings`, `codebase_symbols`, and the structural
edges `codebase_defines_edges`, `codebase_imports_edges`, `codebase_calls_edges`.

You author the **foundation half** with `schema apply`:

```yaml
collections:
  - { name: axioms,            type: document }   # the IS / IS-NOT gate
  - { name: smell_specs,       type: document }   # good/bad smells
  - { name: smell_axiom_edges, type: edge }       # smell  -> axiom it enforces
  - { name: compliance_edges,  type: edge }       # code   -> smell (rel: complies|violates)
  - { name: anchored_by,       type: edge }       # code   -> the node it embodies
  - { name: hanging_chad,      type: document }   # recorded orphans / divergences
  - { name: chad_origin,       type: edge }       # chad   -> the code/node it came from
  - { name: supersedes,        type: edge }       # retirement: new -> retired (de-ratification)
edge_definitions:
  - { name: smell_axiom_edges, from_collections: [smell_specs],    to_collections: [axioms] }
  - { name: compliance_edges,  from_collections: [codebase_files], to_collections: [smell_specs] }
  - { name: anchored_by,       from_collections: [codebase_files], to_collections: [axioms, smell_specs] }
  - { name: chad_origin,       from_collections: [hanging_chad],   to_collections: [codebase_files] }
  - { name: supersedes,        from_collections: [axioms],         to_collections: [axioms] }
named_graphs:
  - { name: method_graph, edges: [smell_axiom_edges, compliance_edges, anchored_by, chad_origin, supersedes] }
```

> `bident_burn` already scaffolds most of these (`axioms`, `smell_specs`,
> `compliance_edges`, `anchored_by`, `hanging_chad`/`chad_origin`, `supersedes`),
> so on HADES-Burn itself you are *populating*, not creating.

Axiom document shape (as `bident_burn` already stores them):

```json
{ "_key": "errors-as-data",
  "name": "Errors Are Data, Not Control Flow",
  "is":     ["Errors propagate via Result/Either", "Failures named by semantic kind"],
  "is_not": ["Bare panics or unwrap() in library code", "Silent error swallowing"],
  "source": { "quote": "<verbatim>", "ratified_by": ["todd"], "ratified": "2026-06-09" } }
```

---

## The structural backbone — directory hierarchy

The graph has **two layers of connectivity**, doing different jobs:

- **The semantic layer** (`imports`, `defines`, `complies`, `embodies`) connects
  by *meaning*. It is **optional and earned** — this is where a node proves its
  place against the gate.
- **The structural backbone** (directory containment) connects by *construction*.
  It is **mandatory and free**. Every file has one parent directory, every
  directory a parent, up to the repo root — so the filesystem is a connected graph
  by definition. Add directory nodes and containment edges and **no node is ever
  structurally isolated**: any two files share a lowest-common-ancestor directory.

This sharpens "orphan." With the backbone in place, an orphan is **never** a
graph-disconnected node — the directory always connects it. An orphan is a node
with **no embodiment bridge to the gate**, despite being structurally present.
Structural isolation becomes impossible; the suspect set goes back to being purely
about *earned* connection.

The backbone also makes inheritance concrete. *Details inherit their parent's
verdict* stops being abstract: a **directory embodies a spec**, and a file
inherits through `file → contained-by → directory → embodies → spec → axiom`. You
bridge a handful of meaningful directories instead of grinding every file, and a
file with no spec of its own still traces to one.

> **It is a traceability overlay, not a training signal** (corrected by an
> empirical run, 2026-06-09). Injected as *training* edges, the directory tree is
> a hub that oversmooths siblings into one collapsed vector. So **exclude
> `codebase_contains_edges` from `relation_order`**, and encode hierarchy/position
> (path, module, crate) as node **features** for learning instead. See
> [`structural-embedding-lessons.md`](structural-embedding-lessons.md), Mode 1 —
> connectivity-for-querying and structure-for-learning are opposite requirements.

**HADES gap (confirmed).** `codebase ingest` builds only the four semantic edge
types; the directory tree above the file is unmodeled. Until ingest builds it
natively, materialize it from the `rel_path` every file already carries:

- `codebase_directories` (document) — one node per directory, `_key` = rel_path
  with `/` → `_`.
- `codebase_contains_edges` (edge) — `directory → file | directory`.

Keep these in the graph for traversal, the zero-orphan guarantee, and
directory-level inheritance — but **exclude `codebase_contains_edges` from
`relation_order`** (the training scope). **Build the backbone first** — it connects
every file with zero orphans before any embodiment work, and it earns its keep at
query time, not training time.

---

## Greenfield — foundation first

```bash
DB=myproject_burn
hades --db _system db create-database $DB 2>/dev/null            # provision (method-owned, never prod)
hades --db $DB schema apply schema.yaml --dry-run 2>/dev/null    # validate
hades --db $DB schema apply schema.yaml -y 2>/dev/null           # apply
```

1–2. **Dissect + ratify.** I read the founding material and draft candidate axioms
   to a file. **You ratify.** Only then does the write run:
   ```bash
   hades --db $DB db insert axioms --data "$(cat ratified-axioms.json)" 2>/dev/null
   ```
3. **Derive smells** → `db insert smell_specs …`, each with `axiom_basis`,
   `polarity: good|bad`, and a `pattern`; wire `smell_axiom_edges`. Cut any smell
   that doesn't trace to a ratified axiom.
4. *(provision + schema — done above)*
5. **Centralize the types** (where the language allows) before any code moves — so
   the contract rides the type, not the file location, and a rule like "only the
   harness touches the DB" can become a boundary the compiler refuses to break.
6. **Ingest, foundation then code — separate passes:**
   ```bash
   hades --db $DB ingest docs/*.md 2>/dev/null                          # foundation
   hades --db $DB codebase ingest . --unparsed-ext cu,cuh 2>/dev/null   # code (its own pass!)
   ```
   Ingest at the **repo root** (file-key = rel_path, `.`/`/` → `_`). A run that
   skips the code pass looks complete and is not.
7. **Bridge + run the suspect set** — build `compliance_edges` (code → smell), then
   the absence queries below. The output is the work-list.
8. **Lay the trace** — generate `anchored_by` edges, then the `//!` docstring
   anchors **from the graph** (never hand-authored, so code and graph can't
   diverge). Only load-bearing files; glue and re-exports inherit their module's.

---

## Brownfield — code first

Ingest the code, reconstruct the gate by reconciliation. **Validate on one
sharply-bounded crate first** — for HADES-Burn that's `hades-prefetch` (4 files,
clean organ; `hades-proto` is too thin to test the loop). Pre-register: flagged
orphans all real, and a manual read finds none missed. Then the four-cell audit,
each cell citing the exact module:

| | should be | should not be |
|---|---|---|
| **implemented** | **A. Keep** → ratify (becomes the gate) | **B. Divergent** → de-ratify or fix |
| **not implemented** | **D. Gap** → backlog | **C. Abandoned** → close it, keep the *why* |

**Companion directory — keep HADES artifacts out of the upstream repo.** When
you're brownfielding a project you intend to contribute to upstream, the artifacts
*you* author for HADES — reconstructed PRDs and specs, the divergence register,
the method schema — are **not** part of the upstream project and must not live in
its repo. Put them in a sibling **companion directory** named `<repo>-hades`
(e.g. `~/git/candle` → `~/git/candle-hades`). Ingest the code from the real repo
and the foundation from the companion dir:

```bash
hades --db $DB codebase ingest ~/git/candle --unparsed-ext cu,cuh 2>/dev/null   # upstream code
hades --db $DB ingest ~/git/candle-hades/docs/*.md 2>/dev/null                  # your reconstruction
```

This keeps your reconstruction out of every PR you send upstream, and avoids a
`.gitignore` entry that would itself get pushed upstream.

---

## The suspect-set queries (read-only `db aql`)

```bash
# Orphan code — files that comply-with / violate nothing (embody nothing documented)
hades --db $DB db aql 'FOR f IN codebase_files
  LET n = LENGTH(FOR e IN compliance_edges FILTER e._from == f._id LIMIT 1 RETURN 1)
  FILTER n == 0 RETURN f._key' 2>/dev/null | jq .data

# Violations — code that breaks a smell
hades --db $DB db aql 'FOR e IN compliance_edges FILTER e.rel == "violates"
  RETURN { file: e._from, smell: e._to }' 2>/dev/null | jq .data

# Unembodied — smells nothing complies with (a rule no code satisfies)
hades --db $DB db aql 'FOR s IN smell_specs
  LET n = LENGTH(FOR e IN compliance_edges FILTER e._to == s._id AND e.rel == "complies" LIMIT 1 RETURN 1)
  FILTER n == 0 RETURN s._key' 2>/dev/null | jq .data
```

**An orphan is a question, not a verdict** — code with no bridge resolves four
ways: real-but-undocumented (write the spec), wrong-organ (move it), vestigial
(delete or justify), or a **tacit invariant** (articulate it — the valuable case a
delete-on-sight rule destroys). **Coverage is per-node:** check each load-bearing
feature has an anchor, never an aggregate annotated-file percentage.

---

## Enforcement — violation edges

A code node has **three states**, not two:

- **Compliant** — a `compliance_edges` edge with `rel: complies`, tracing through
  the smell to a ratified axiom.
- **Violating** — a `compliance_edges` edge with `rel: violates`, naming the smell
  (and through it the axiom) it breaks. Present, but **negatively connected**.
- **Orphan** — no bridge edge at all (still hangs on the directory backbone, but
  embodies nothing documented).

Violation is the **presence of a negative edge, not absence** — don't refuse a
violating node, *record* it. A rejected node is an *invisible* violation: the
moment the graph won't admit it, the break disappears, which is the opposite of
what the immune system is for. Joining it with a `violates` edge makes the break
auditable.

The violation edge carries a **verdict**: `unratified` (a real suspect) or
`ratified-exception` (a known, why-documented divergence — e.g. a fork's
deliberate patch). The same edge type records both the break and the permission;
enforcement acts only on the unratified ones.

```bash
# Enforcement query — every unratified break
hades --db $DB db aql 'FOR e IN compliance_edges
  FILTER e.rel == "violates" AND e.verdict == "unratified"
  RETURN { file: e._from, smell: e._to, evidence: e.evidence }' 2>/dev/null | jq .data
```

Materialize a violation edge for **both** polarities: a bad smell when its
forbidden pattern *matches*, a good smell when its required pattern is *absent*
(an omission) — so enforcement stays one uniform query. **Record and enforce are
two layers:** the graph records the edge; a CI/pre-merge gate enforces by failing
when that query is non-empty. The strongest enforcement isn't the graph at all —
structural breaks (boundary violations) belong to the compiler, which catches them
before ingest; the violation-edge layer is for the *semantic* breaks the compiler
can't see (an `unwrap()` in library code compiles fine but violates
`errors-as-data`).

---

## De-ratification — the exit gate

When reality falsifies a ratified axiom, **demote, don't delete**, and re-open its
dependents:

```bash
hades --db $DB db update axioms <old> --data '{"verdict":"retired","retired":"<date> — <why>"}' 2>/dev/null
hades --db $DB db insert supersedes --data '[{"_from":"axioms/<new>","_to":"axioms/<old>"}]' 2>/dev/null
# then re-run the suspect set: everything that bridged to <old> is now pending re-ratification
```

The `supersedes` edge keeps the *why of the retraction* in the graph. A gate with
no exit is a prophecy that fulfills itself.

---

## The human gate

I draft candidates (axioms, smells, decisions) to a file or to stdout and **stop**.
You ratify or rewrite. Only then does the `db insert` run. When the build meets
reality and they disagree, **you rule** — amend the canon or reject the build — and
I record the friction. The only writable target is your method DB; production
stays read-only by ACL, the backstop if I get it wrong.

---

## The provenance ritual (work tracking without a second store)

There is no kanban, no task mirror, no sync. Work tracking is a citation ritual
over instruments that already exist, at two resolutions:

- **Public, coarse:** work starts as an **issue** (intent and adjudication,
  even one paragraph). The **PR** carries the change, and review approval is
  the ratification. The **squash merge stamps `(#N)`** into permanent local
  history.
- **Local, fine:** the **PRD** holds requirements and the why, the **Spec**
  holds the architecture, both as repo files (ingestable under the standard
  contract, nothing special).
- **Stitching is citations only:** the issue names the PRD path, the PR body
  cites the Spec and `Closes #N`, the merge stamps the number. The whole chain
  `git log -> PR -> issue -> PRD -> Spec` is then recoverable with zero
  standing machinery.

The rule that keeps this honest: **ingest facts that cannot change, read live
anything that can.** Merged PRs and commits are append-only history, safe to
index (the commit-to-PR half is derivable offline from `git log` thanks to the
`(#N)` convention). Open-issue state is mutable and never enters the graph:
query `gh` for it. A synced mirror of mutable forge state is a second source
of truth and a standing tax, which is how the old kanban died.

If discipline slips, enforcement is earned by the breach, like smell
promotion: a small CI lint (any merge without `(#N)`, any PR without an
issue), absence queries over immutable history. Build it then, not before.

---

## Structural embeddings (optional, HADES-specific)

Once code is ingested + embedded and the graph is stable:

```bash
hades --db $DB --gpu 1 graph-embed train 2>/dev/null   # model_type from schema (rgcn | hetero_sage)
```

Needs the training service up; `--checkpoint-dir` must be writable by **both** you
and the `hades` user (the CLI runs as you, the trainer as `hades`).

**`--gpu N` is mandatory for `graph-embed train` and `graph-embed update`** (PR
#143): training has no default device anywhere in the chain, by design. GNN
training is infrequent, and an infrequently exercised default is a forgotten
default. Declare the card per run. The embedder keeps its default because it
runs constantly: frequency decides whether a default is safe.

**Embedding quality is governed by the graph's relational design, not the
pipeline.** Several modeling errors each independently cause *total collapse*
(every node → one vector). Doctrine:

- **Hierarchy/position → node features, not edges.** Containment/membership edges
  are a traceability overlay; **exclude them from `relation_order`** (hub
  oversmoothing).
- **Curate `relation_order` for signal balance, not completeness.** A relation
  that's >~90% of edges captures the objective and collapses everything.
- **Make semantic edges bidirectional** (or add reverse relations) so leaf/code
  node types *receive* inflow, and make sure that inflow carries information the
  target does not already have. A sink with all-outgoing edges cannot learn, and
  neither can a node whose only in-neighbors echo its own feature back at it
  (self-referential inflow is no inflow).
- **Featurize every participating node type** — a featureless type is silently
  skipped (verify `0 embeddings skipped`).
- **Diagnose with per-node-type embedding diversity** (cosine ≈ 1.0 = collapse),
  never aggregate AUC alone.

Full evidence trail, the five collapse modes, the operational gotchas, and the
schema-lint / post-train-report tooling recommendations:
[`structural-embedding-lessons.md`](structural-embedding-lessons.md).

---

## Reliability checklist

- `--db` set and `2>/dev/null` appended on every call.
- `codebase ingest` at the **repo root**; the code ingest is its **own pass**.
- `db aql` is read-only; writes go through structured `db` ops.
- Re-ingest purges + rebuilds changed files; unchanged files are skipped.
- Only the method DB is writable; production is ACL read-only.
