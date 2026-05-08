# Design: RACE — Retrieval-Augmented Context Engineering

**Status**: Draft / Request for Discussion
**Date**: 2026-05-08 (revised)
**Scope**: HADES-Burn — RACE positioning, declarative schema (`hades schema apply`),
domain-agnostic RACE primitives, worked code demo
**Author**: Todd Bucy + Claude (HADES-Burn session)

---

## 0. What HADES Is — RACE

HADES is a **Retrieval-Augmented Context Engineering** platform built on
ArangoDB. RACE is a meaningful step beyond RAG:

| | RAG | RACE |
|---|---|---|
| **What's retrieved** | Relevant documents | Schema constraints, axioms, compliance edges, *and* relevant documents |
| **What's done with the retrieval** | Inserted into the model's context for generation | Used to *bound* what concepts the agent is allowed to think about — the schema acts as an immune system rejecting concepts not in the ontology |
| **Where the constraint lives** | Implicit in the prompt | Explicit in the graph (queryable, versionable, auditable) |
| **Failure mode if absent** | Hallucinated facts | Hallucinated *concepts* — agent invents structure that contradicts the ontology |

HADES has three layers, in dependency order:

```text
┌─────────────────────────────────────────────────────┐
│  RACE Layer — the product                           │
│  · Schema-as-constraint (IS / IS NOT axioms)        │
│  · Compliance edges (graph-as-immune-system)        │
│  · Smell-check as a worked example                  │
│  · Configured per-domain via YAML                   │
├─────────────────────────────────────────────────────┤
│  Knowledge Graph Layer                              │
│  · Codebase universal ontology                      │
│  · Vector search + hybrid + structural rerank       │
│  · Schema-as-data (`hades_schema` collection)       │
├─────────────────────────────────────────────────────┤
│  DBA Layer — the substrate                          │
│  · CRUD, AQL, indexes, named graphs                 │
│  · Backups, exports, schema apply                   │
└─────────────────────────────────────────────────────┘
```

The **substrate alone is marginal value over raw AQL** — a competent DBA
can write the same queries by hand. The *product* is the upper layer:
the graph constrains agent behavior because the schema is queryable
data, not prose in a system prompt.

This document specifies the mechanism that makes the upper layer
configurable per-domain — declarative YAML schemas applied at bootstrap.

## 1. Problem

After the strangler-fig pass (PRs #73 → #80), HADES-Burn is a generic graph
framework. The schema for any given database is supposed to be *data* in
the `hades_schema` collection — not code. But getting that data *into* a
fresh database is currently:

- **Possible only via `--seed empty`** (zero domain content) or hand-rolled
  AQL inserts.
- **Inconsistent with the RACE primitives**, which still have NL-specific
  hardcoded bits leftover from the original Nested-Learning prototype
  (`STATIC_SMELL_IDS = &[10, 11, 13, 40]` in `dispatch.rs:4499`, the
  literal collection name `nl_smell_compliance_edges`, no seeding
  mechanism for `smell_specs`). The *machinery is the product*; the
  problem is that it's not yet domain-agnostic.
- **Not version-controllable**: a project's schema lives only in the live
  database, can't be checked into git, can't be shared, can't be re-applied.

The CLI audit (`scripts/cli_audit.sh`) made the gap concrete:

> `smell report` errors with `failed to load smell definitions` on a
> fresh database, and there's no documented way to seed `smell_specs`.

This document proposes a **declarative schema-as-YAML** mechanism with a
specific stance on what's the source of truth, and when.

## 2. Core Principle — Dual Source of Truth

> **At bootstrap time, the YAML is canonical.**
> **Once a database is in use, the database is canonical and the YAML
> becomes a configuration artifact.**

This single distinction shapes the rest of the design.

| Phase | Source of truth | YAML's role | Mutation channel |
|---|---|---|---|
| **Bootstrap** (fresh DB) | YAML | Canonical seed | `hades schema apply <file>` |
| **Runtime** (in-use DB) | Database | Snapshot / reference | Direct CRUD: `db insert`, agent writes, `db schema add-edge` |
| **Capture** (snapshot) | Database | Updated artifact | `hades schema export > file.yaml` |

The YAML is not a Rails-style migration history. There is no
`002_add_axioms.yaml`. There is one file (or a small number of files), and
it represents *the seed* — what the database started as. After that, the
database evolves independently. If you want a current snapshot, you
`export`. If you want to share a starting point for a new project, you
hand someone the seed YAML.

## 3. Lifecycle

### 3.1 Bootstrap workflow (new project)

```text
Author schema.yaml
   ↓
hades db create-database my_project
   ↓
hades --db my_project schema apply schema.yaml
   ↓
Project's database now has:
  - User-defined collections (axioms, smell_specs, etc.)
  - User-defined documents (the actual axiom values, smell rules)
  - User-defined edge definitions registered in hades_schema
  - User-defined named graphs registered in hades_schema
```

### 3.2 Evolve workflow (in-use project)

```text
hades --db my_project db insert axioms --data '<new-axiom-doc>'   # new content
hades --db my_project db schema add-edge <new-edge-def>           # new edge type
# (or) an agent calls hades --db my_project db insert / db update via its tools
   ↓
Database has evolved beyond what schema.yaml describes.
This is expected. The YAML is now stale relative to the DB.
```

### 3.3 Snapshot workflow (capture state)

```text
hades --db my_project schema export > schema.yaml
   ↓
schema.yaml now reflects current DB state.
Commit to git for history / sharing.
```

## 4. CLI Surface

```text
hades --db <name> schema apply <file>           # bootstrap (refuses if DB in-use)
hades --db <name> schema apply <file> --dry-run # show plan without executing
hades --db <name> schema apply <file> --force   # apply even to in-use DB (dangerous)
hades --db <name> schema export                 # write current DB schema to stdout
hades --db <name> schema diff <file>            # show drift between file and DB
```

`--db` is mandatory (per the post-#80 policy; no implicit defaults).

### 4.1 "In-use" detection heuristic

`schema apply` to an in-use DB without `--force` should refuse. The
heuristic for "in-use":

- `hades_schema` collection exists AND contains documents beyond the
  default `meta` document, OR
- Any user-defined collection (per the schema in the file) exists and
  contains > 0 documents.

The codebase universal layer (`codebase_files`, etc.) does **not** count
as "in-use" — those collections may exist from prior `codebase ingest`
runs and shouldn't block schema bootstrap.

## 5. YAML Schema

### 5.1 Structure

```yaml
# Top-level keys:
collections:        # required collections (document or edge type)
edge_definitions:   # registered in hades_schema
named_graphs:       # registered in hades_schema (gharial)
<user_collection>:  # documents to seed per collection (e.g., axioms:, smell_specs:)
```

### 5.2 Concrete example

```yaml
# my-project/schema.yaml — context-engineering bootstrap

collections:
  - name: axioms
    type: document
  - name: smell_specs
    type: document
  - name: compliance_edges
    type: edge

axioms:
  - _key: container-is
    name: "Nested-Learning Container"
    is:
      - "nested multi-level optimization"
      - "self-modifying components"
    is_not:
      - "single-level optimization"
      - "external optimizer"
      - "train/eval distinction"

smell_specs:
  - _key: smell-010
    tier: static
    pattern: 'model\.eval\(\)'
    description: "NL has no train/eval distinction"
  - _key: smell-011
    tier: static
    pattern: 'optim\.SGD|optim\.Adam'
    description: "NL has no external optimizer"

edge_definitions:
  - name: compliance_edges
    source_field: source
    from_collections: [codebase_files, codebase_symbols]
    to_collections:   [axioms, smell_specs]
    description: "Code → axiom/smell compliance"

named_graphs:
  - name: context_compliance
    edges: [compliance_edges]
    description: "Traverse code → axioms/smells"
```

### 5.3 Vocabulary: fixed or open-ended?

**Open-ended.** The YAML allows arbitrary top-level collection names
(`axioms`, `smell_specs`, `protein_residues`, whatever the user defines).
The translator looks up each top-level key in the `collections:` section
and emits inserts into the corresponding collection.

This means HADES doesn't bake in "axioms" or "smells" as concepts at the
schema-application level. They're conventions, not requirements. The
generalization step in §9 strips NL-specific naming so that *commands
like `smell check`* read the convention from the data, not from
hardcoded constants.

## 6. Internal Pipeline (Generated, Not User-Authored)

```text
schema.yaml
   ↓
[1. parse]      serde-deserialize into a Rust struct (validates structure)
   ↓
[2. validate]   semantic checks: edge from/to collections must be declared,
                document _keys must be unique within a collection, types
                must match (e.g., edge collections receive edge documents)
   ↓
[3. plan]       emit ordered list of operations:
                 a. POST /_api/collection for each missing collection
                 b. AQL UPSERTs for each document, batched by collection
                 c. UPSERTs into hades_schema for edge_definitions and named_graphs
                 d. POST /_api/gharial for each named graph
   ↓
[4. dry-run?]   if --dry-run: print plan as JSON, exit
   ↓
[5. apply]      execute plan, log per-operation results
   ↓
[6. verify]     re-query the DB; assert all expected documents exist
                with the expected values
```

### 6.1 Generated AQL (illustrative, never authored by user)

```aql
// Step 5 (apply, per axiom document):
UPSERT { _key: "container-is" }
  INSERT { _key: "container-is", name: "...", is: [...], is_not: [...] }
  UPDATE { name: "...", is: [...], is_not: [...] }
IN axioms

// Step 5 (apply, per smell_spec):
UPSERT { _key: "smell-010" }
  INSERT { _key: "smell-010", tier: "static", pattern: "...", description: "..." }
  UPDATE { tier: "static", pattern: "...", description: "..." }
IN smell_specs

// Step 5 (apply, edge definition into hades_schema):
UPSERT { _key: "edge__compliance_edges__source" }
  INSERT { _key: "edge__compliance_edges__source",
           schema_type: "edge_definition",
           name: "compliance_edges",
           ... }
  UPDATE { schema_type: "edge_definition", name: "compliance_edges", ... }
IN hades_schema
```

UPSERT semantics make the file idempotent — running `schema apply` twice
on a fresh DB produces the same result as running it once. The
distinction between "fresh" and "in-use" is enforced before the apply
runs, not by relying on UPSERT to be safe.

## 7. Idempotency Model

| Operation | Strategy |
|---|---|
| Collection creation | Ignore "duplicate name" errors; check type matches if exists |
| Document insert | UPSERT by `_key`, INSERT branch + UPDATE branch with same fields |
| Edge definition registration | UPSERT into `hades_schema` |
| Named graph creation | Ignore "graph exists" errors; check edge defs match |

Re-running `schema apply` on a database it already bootstrapped is a
no-op (in spirit; some UPDATEs may touch timestamps but content stays
the same). Re-running with a *modified* YAML on an in-use DB is what
`--force` exists for — and is documented as dangerous.

## 8. Validation

Two layers:

1. **Structural**: serde deserialization rejects missing required
   fields with clear "field X missing at line N" errors. Note that
   serde's *default* behavior silently ignores unknown fields, which
   would let typos in user-authored YAML pass without warning. The
   implementation MUST annotate every schema struct with
   `#[serde(deny_unknown_fields)]` (or perform an equivalent runtime
   check) so that `axiom:` written instead of `axioms:` is caught at
   parse time rather than silently dropped.
2. **Semantic**: after parse, the validator checks:
   - Every edge collection in `edge_definitions[].from_collections` /
     `to_collections` is also declared in `collections:`
   - Every collection referenced by per-collection document blocks
     (`axioms:`, `smell_specs:`) is declared in `collections:`
   - `_key` values are unique within each collection
   - `collections[].type` is `document` or `edge`

Validation runs **before** any DB write. A malformed file never touches
the database.

## 9. Making the RACE Primitives Domain-Agnostic

The RACE primitives — axioms, smell_specs, compliance edges,
smell-check command — exist today but carry NL-specific residue from
the original prototype. The work isn't to *generalize* a code-specific
feature into a generic one; it's to **strip the residue so the
primitives reveal their domain-agnostic shape**.

| Currently hardcoded | Becomes |
|---|---|
| `STATIC_SMELL_IDS = &[10, 11, 13, 40]` (`dispatch.rs:4499`) | `tier` field on each `smell_spec` document |
| `BEHAVIORAL_SMELL_IDS = &[27, 28, 32]` | Same — `tier` field |
| `ARCHITECTURAL_SMELL_IDS = &[31]` | Same — `tier` field |
| `nl_smell_compliance_edges` literal in AQL | `compliance_edges` (collection name read from `hades_schema`) |
| `nl_code_smells` literal in AQL | `smell_specs` (collection name read from `hades_schema`) |
| `arxiv_metadata_*__nl_code_smells_*` composite key formula | Generic `<source_collection>_<source_key>__<spec_collection>_<spec_key>` |
| `smell-NNN-` key prefix convention | Documented as convention; not enforced in code |

The smell-check command itself stays — it's not application logic
intruding on a database tool, it's the primary RACE primitive. What
makes it domain-agnostic after this refactor:

- The patterns to match come from `smell_specs` documents (already
  data-driven; just need to remove the assumption that the collection
  is called `nl_code_smells`)
- The tier classification comes from the document's `tier` field,
  not a hardcoded ID lookup
- The compliance edges land in whatever edge collection the user
  declared in their YAML (typically `compliance_edges`)
- The "scan files from disk" part remains code-specific because *files
  on disk are inherently the code domain* — other domains (memory DBs,
  paper corpora) would author their own scanners that compose HADES
  primitives, and HADES doesn't pretend they're the same as code

This is a refactor of comparable scope to PR #77 (the NL_GRAPH_SCHEMA
migration). It's the prerequisite for `schema apply` to bootstrap a
working RACE setup end-to-end.

## 10. The Code Demo: `seeds/code-context-engineering.yaml`

The code-context-engineering YAML is the framework's **flagship
demonstration** of RACE. It exists for three reasons:

1. **Code is universal.** Every project has source code; every agent
   interacting with HADES will encounter it. Code is the domain that
   most validates "does the framework actually work end-to-end?"
2. **It's the most complete worked example.** The codebase universal
   ontology already exists. Adding RACE-layer constraints (axioms,
   smells, compliance edges) on top showcases the whole stack.
3. **It defines the convention other domains will follow.** Memory
   DBs, paper corpora, and protein-folding databases will all want
   their own constraint sets. The code demo establishes the *shape*
   of that authoring.

```yaml
# seeds/code-context-engineering.yaml
#
# Worked example: turn any HADES instance into a code-context-engineering
# platform. Works against the universal codebase ontology
# (codebase_files, codebase_symbols, codebase_chunks).

collections:
  - { name: axioms, type: document }
  - { name: smell_specs, type: document }
  - { name: compliance_edges, type: edge }

axioms:
  - _key: testable-functions
    name: "Testable Functions"
    is:
      - "Pure functions with explicit inputs and outputs"
      - "Side effects isolated to specific layers"
    is_not:
      - "Hidden global state mutations"
      - "Implicit time/random dependencies"

  - _key: error-propagation
    name: "Errors Are Data, Not Control Flow"
    is:
      - "Errors propagate via Result types"
      - "Failures named by their semantic kind"
    is_not:
      - "Silent error swallowing"
      - "Bare panics in library code"

smell_specs:
  - _key: smell-010
    tier: static
    pattern: 'unwrap\(\)\s*$'
    description: "Bare .unwrap() outside tests"
    blocks_ingest: true
    related_axiom: error-propagation

  - _key: smell-020
    tier: behavioral
    pattern: 'TODO|FIXME|XXX'
    description: "Unfinished work markers"
    blocks_ingest: false

edge_definitions:
  - name: compliance_edges
    from_collections: [codebase_files, codebase_symbols]
    to_collections:   [axioms, smell_specs]
    description: "Code → axiom/smell compliance"

named_graphs:
  - name: code_context_compliance
    edges: [compliance_edges, codebase_defines_edges, codebase_calls_edges]
    description: "Code structure + compliance constraints in one named graph"
```

**Bootstrap workflow with the demo:**

```bash
hades db create-database my_codebase
hades --db my_codebase schema apply seeds/code-context-engineering.yaml
hades --db my_codebase codebase ingest .
hades --db my_codebase smell check src/
```

After step 4, every file/symbol with a smell violation has a
`compliance_edges` document linking it to the violated `smell_spec`.
The agent reading this graph can `db query` for "code violating axiom
'error-propagation'" and get a structured answer it can act on.

**This is the killer demo.** It's how a contributor or evaluator sees
HADES go from "graph database with extras" to "context-engineering
platform that actually constrains model behavior."

## 11. Out of Scope (Intentional)

- **Schema migrations**: renames, type changes, deletions. Once a DB is
  in use, the YAML is *not* the way to evolve it. Use direct CRUD or
  write a one-off AQL script.
- **Multi-file schemas / includes**: one file per project. If you want
  shared building blocks, copy/paste into your file or use a templating
  tool outside HADES.
- **Bidirectional sync**: `apply` and `export` are not transactional
  inverses. The DB may have data the YAML doesn't and vice versa.
  Drift is expected.
- **Rollback**: no `schema rollback` command. If `apply` fails partway,
  the state is whatever ArangoDB left it in (collection-creation is
  idempotent, AQL UPSERTs are per-document atomic). Recovery is manual
  or via `db drop-database` + re-apply.

## 12. Open Questions

The following items track open questions and their current status.
Items marked **(settled, &lt;date&gt;)** have been finalized and are
recorded here for the design rationale; remaining items still need
calls before implementation begins.

- **Q1**: What does `schema apply --force` actually do on an in-use DB?
  Replace conflicting documents? Refuse to delete? Add new ones? Most
  conservative: only INSERT new documents and edge_definitions; never
  UPDATE or DELETE. Most powerful: full UPSERT semantics across
  everything.
- **Q2**: Where do validators live for project-defined collections?
  E.g., a project's YAML declares an `axioms` collection — does HADES
  need a built-in JSON schema for what an `axiom` document looks like?
  Or is the user's YAML the only contract and HADES just inserts whatever?
  Probably: HADES validates the YAML *structure* (per §8), not the
  *semantics* of user-defined documents. Domain-specific validation is
  the user's responsibility.
- **Q3** (settled, 2026-05-08): Yes — the framework ships
  `seeds/code-context-engineering.yaml` as the canonical demo (see
  §10). Other domain seeds (memory DBs, paper corpora) are deferred
  until there's a concrete consumer asking for them. Starter YAMLs
  are referenced by path, not by an opaque `--template` flag — keeps
  the file as the artifact, the flag would just hide it.
- **Q4**: Naming: `schema apply` vs `project apply` vs `bootstrap` —
  this is mostly bikeshed but will appear everywhere in docs and tool
  call signatures. Recommendation: **`schema apply`** (matches kubectl
  and existing `db schema init` neighbors).
- **Q5**: Does `--seed empty` survive after `schema apply` lands? Or
  does it become an alias for `schema apply <ship-an-empty-yaml>`?
  Cleaner: deprecate `db schema init --seed` once `schema apply` covers
  the same ground.

## 13. Rollout Plan

Four PRs in sequence:

1. **PR α — Make RACE primitives domain-agnostic.**
   - Strip `nl_smell_compliance_edges` literal → `compliance_edges`
     (read collection name from `hades_schema`)
   - Strip `nl_code_smells` literal → `smell_specs`
   - Move smell tier IDs from `STATIC_SMELL_IDS` constants to per-document
     `tier` fields; update the smell handlers to read from documents
   - Generalize the composite key formula
     (`arxiv_metadata_*__nl_code_smells_*` → generic
     `<source_collection>_<source_key>__<spec_collection>_<spec_key>`)
   - No new commands; the existing `smell check / verify / report /
     link` commands become domain-agnostic

2. **PR β — `hades schema apply`.**
   - Define YAML struct (Rust serde-Deserialize, all
     `#[serde(deny_unknown_fields)]`)
   - Validator (structural + semantic)
   - Planner (emit operation list)
   - Applier (execute plan, idempotent)
   - `--dry-run` and `--force` flags
   - Tests with sample YAMLs

3. **PR δ — Ship `seeds/code-context-engineering.yaml`.**
   - The flagship RACE demo (per §10)
   - Documents the convention other domains will follow
   - Includes a `README.md` next to it with the bootstrap workflow
   - Optionally: a smoke-test script that runs the full sequence
     (`schema apply` → `codebase ingest` → `smell check`) and
     verifies compliance edges land

4. **PR γ — `schema export` + `schema diff`** (optional, defer until needed).
   - Inverse of `apply`
   - Exports current DB state to YAML
   - `diff` shows drift

**Critical path**: PR α → PR β → PR δ. PR γ is operational ergonomics
that can wait. After PR δ, HADES has a working RACE-against-code
demonstration any user or evaluator can reproduce in five commands.

## 14. Connects To

- [Schema as Ontology](memory: project_schema_as_ontology.md) — this is
  the practical end of the data-driven schema architecture
- [No Raw AQL for Models](memory: feedback_no_raw_aql.md) — YAML is the
  model-friendly authoring format; AQL is internal
- [Design for Training Distribution](memory: feedback_model_training_distribution.md)
  — YAML structure can be tuned to match what 24-32B models expect
- [Agent Memory Architecture](docs/design-agent-memory-and-system-prompt.md)
  — `schema apply` is part of how agent memory DBs get provisioned

---

## Appendix A — Comparison with Alternatives Considered

| Approach | Why not |
|---|---|
| **AQL files** (raw) | Pure AQL can't create collections; would need preprocessor anyway |
| **Migration-numbered files** (Rails-style) | Adds ordering complexity; YAML is bootstrap-only so unnecessary |
| **Programmatic API** (Rust crate users build their own seeders) | Excludes non-Rust users; doesn't version-control nicely |
| **Direct `db insert` + `db schema add-edge`** | What we have today — laborious, error-prone, no atomicity |

## Appendix B — Example: Onboarding a New Project

```bash
# 1. Author the schema (one-time, version-controlled)
$ cat > my-project/schema.yaml <<EOF
collections:
  - { name: my_concepts, type: document }
my_concepts:
  - _key: foo
    description: "first concept"
EOF

# 2. Provision the database. `schema apply` is the single bootstrap
#    entrypoint: it creates `hades_schema` (if missing) along with every
#    collection declared in the YAML, then populates them. There is no
#    separate `db schema init` step required.
$ hades db create-database my_project_db
$ hades --db my_project_db schema apply my-project/schema.yaml

# 3. Verify
$ hades --db my_project_db db count my_concepts
{"count": 1}

# 4. Now use it normally — DB is the source of truth from here on
$ hades --db my_project_db db insert my_concepts --data '{"_key":"bar","description":"second"}'
$ hades --db my_project_db db count my_concepts
{"count": 2}

# 5. Capture current state back to YAML (optional)
$ hades --db my_project_db schema export > my-project/snapshot.yaml
```
