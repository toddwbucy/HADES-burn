# Design: Declarative Schema (YAML at Bootstrap, DB at Runtime)

**Status**: Draft / Request for Discussion
**Date**: 2026-05-08
**Scope**: HADES-Burn — `hades schema apply`, context-engineering generalization
**Author**: Todd Bucy + Claude (HADES-Burn session)

---

## 1. Problem

After the strangler-fig pass (PRs #73 → #80), HADES-Burn is a generic graph
framework. The schema for any given database is supposed to be *data* in
the `hades_schema` collection — not code. But getting that data *into* a
fresh database is currently:

- **Possible only via `--seed empty`** (zero domain content) or hand-rolled
  AQL inserts.
- **Inconsistent with the smell/context-engineering machinery**, which
  still has hardcoded NL-specific bits (`STATIC_SMELL_IDS = &[10, 11, 13, 40]`
  in `dispatch.rs:4499`, the literal collection name `nl_smell_compliance_edges`,
  no seeding mechanism for `smell_specs`).
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
db insert axioms <new-axiom-doc>           # new content
db schema add-edge <new-edge-def>          # new edge type
agent context-engineers a code change      # writes via existing CRUD
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
generalization step in §10 strips NL-specific naming so that *commands
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
// Step 4 (per axiom document):
UPSERT { _key: "container-is" }
  INSERT { _key: "container-is", name: "...", is: [...], is_not: [...] }
  UPDATE { name: "...", is: [...], is_not: [...] }
IN axioms

// Step 4 (per smell_spec):
UPSERT { _key: "smell-010" }
  INSERT { _key: "smell-010", tier: "static", pattern: "...", description: "..." }
  UPDATE { tier: "static", pattern: "...", description: "..." }
IN smell_specs

// Step 4 (edge definition into hades_schema):
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

1. **Structural**: serde deserialization rejects unknown fields and
   missing required fields. Output: clear "field X missing at line N"
   errors.
2. **Semantic**: after parse, the validator checks:
   - Every edge collection in `edge_definitions[].from_collections` /
     `to_collections` is also declared in `collections:`
   - Every collection referenced by per-collection document blocks
     (`axioms:`, `smell_specs:`) is declared in `collections:`
   - `_key` values are unique within each collection
   - `collections[].type` is `document` or `edge`

Validation runs **before** any DB write. A malformed file never touches
the database.

## 9. Generalizing Smell / Axiom Infrastructure

For schema YAML to be useful for context-engineering, the smell
machinery needs to be generic:

| Currently hardcoded | Becomes |
|---|---|
| `STATIC_SMELL_IDS = &[10, 11, 13, 40]` (`dispatch.rs:4499`) | `tier` field on each `smell_spec` document |
| `BEHAVIORAL_SMELL_IDS = &[27, 28, 32]` | Same — `tier` field |
| `ARCHITECTURAL_SMELL_IDS = &[31]` | Same — `tier` field |
| `nl_smell_compliance_edges` literal in AQL | `compliance_edges` (or read collection name from `hades_schema`) |
| `smell-NNN-` key prefix convention | Documented as convention; not enforced |

This is a separate refactor of comparable scope to PR #77 (the
NL_GRAPH_SCHEMA migration). It's a prerequisite for `schema apply` to
work end-to-end with context-engineering content.

## 10. Out of Scope (Intentional)

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

## 11. Open Questions

These remain undecided and should be settled before implementation
begins:

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
- **Q3**: Should the framework ship "starter" YAMLs (e.g.,
  `seeds/context-engineering.yaml`)? If yes, where do they live and how
  are they referenced (`schema apply --template context-engineering`)?
- **Q4**: Naming: `schema apply` vs `project apply` vs `bootstrap` —
  this is mostly bikeshed but will appear everywhere in docs and tool
  call signatures. Recommendation: **`schema apply`** (matches kubectl
  and existing `db schema init` neighbors).
- **Q5**: Does `--seed empty` survive after `schema apply` lands? Or
  does it become an alias for `schema apply <ship-an-empty-yaml>`?
  Cleaner: deprecate `db schema init --seed` once `schema apply` covers
  the same ground.

## 12. Rollout Plan

Three PRs in sequence:

1. **PR α — Generalize smell/axiom infrastructure.**
   - Strip `nl_smell_compliance_edges` literal → `compliance_edges` (read
     from `hades_schema` if needed)
   - Move smell tier IDs from `STATIC_SMELL_IDS` constants to per-document
     `tier` fields; update the smell handlers to read from documents
   - No new commands; just makes the existing smell commands generic

2. **PR β — `hades schema apply`.**
   - Define YAML struct (Rust serde-Deserialize)
   - Validator (structural + semantic)
   - Planner (emit operation list)
   - Applier (execute plan, idempotent)
   - `--dry-run` and `--force` flags
   - Tests with sample YAMLs

3. **PR γ — `schema export` + `schema diff`** (optional, defer until needed).
   - Inverse of `apply`
   - Exports current DB state to YAML
   - `diff` shows drift

PR α alone fixes the smell infrastructure. PR β alone unblocks
context-engineering bootstrapping. PR γ adds operational ergonomics.

## 13. Connects To

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

# 2. Provision the database
$ hades db create-database my_project_db
$ hades --db my_project_db db schema init --seed empty
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
