# Seeds

Reference YAML schemas used by `hades schema apply` to bootstrap a
database with a particular RACE (Retrieval-Augmented Context
Engineering) constraint set.

A seed file declares:

| Section            | Purpose                                                   |
|--------------------|-----------------------------------------------------------|
| `collections`      | Document and edge collections required by the RACE layer. |
| `<collection>:`    | Per-collection document seeds (top-level YAML key).       |
| `edge_definitions` | from/to constraints for edge collections.                 |
| `named_graphs`     | Gharial named graphs combining edge collections.          |

See `docs/declarative-schema.md` for the full design and
`crates/hades-core/src/schema_apply.rs` for the YAML parser /
validator / applier.

## Available Seeds

### `code-context-engineering.yaml`

The flagship demo. Layers RACE constraints (axioms, smell_specs,
compliance_edges) on top of the universal codebase ontology
(`codebase_files`, `codebase_symbols`, `codebase_*_edges` produced by
`hades codebase ingest`).

What it adds:

- **4 axioms** — IS / IS NOT design rules
  (`testable-functions`, `errors-as-data`, `explicit-dependencies`,
  `data-flow-clarity`).
- **8 smell specifications** spanning all three tiers
  (static / behavioral / architectural), with `forbidden_patterns`
  the runtime `smell_check` recognizes.
- **`compliance_edges`** edge collection linking code → axioms /
  smells.
- **`code_context_compliance`** named graph combining defines, calls,
  and compliance edges into one traversable view.

## Bootstrap Workflow

The full code-context-engineering demo, top to bottom:

```bash
# 1. Create the database.
hades db create-database my_codebase

# 2. Apply the seed: collections, axioms, smells, edge defs, named graph.
hades --db my_codebase schema apply seeds/code-context-engineering.yaml

# 3. Ingest a codebase into the universal ontology.
hades --db my_codebase codebase ingest .

# 4. Run the smell scan; static-tier hits write compliance_edges
#    automatically, behavioral / architectural hits surface as JSON.
hades --db my_codebase smell check src/

# 5. Query — agent reads compliance edges to answer
#    "what code violates errors-as-data?"
hades --db my_codebase db query "code violating errors-as-data"
```

After step 4, every file or symbol with a static-tier violation has a
`compliance_edges` document linking it to the violated `smell_specs`
document. An agent (or a human) asking _why_ a particular smell exists
can traverse one edge in the other direction:
`smell_specs → axioms` is encoded by the `related_axiom` field on
each smell spec.

## Idempotency

`schema apply` is safe to re-run. Existing collections trigger a 409
Conflict that is treated as success; per-document writes use
`onDuplicate=replace`; named graphs already created get the same
treatment. The universal codebase layer is exempt from the
`--force`-required "in-use" guard — you can apply a seed onto a
database that already has `codebase ingest` data.

## Authoring a New Seed

1. Copy `code-context-engineering.yaml` as a starting template.
2. Replace axioms and smells with rules specific to your domain
   (memory DB, paper corpus, protein folding, …).
3. Keep the `compliance_edges` shape — its meaning generalizes:
   a source-of-truth document either satisfies an axiom or
   violates a smell.
4. Validate without writing:
   ```bash
   hades --db my_db schema apply seeds/your-seed.yaml --dry-run
   ```
5. Apply for real once the dry-run plan looks right.

`schema apply --force` overrides the in-use guard for non-universal
collections; reach for it deliberately, not reflexively.
