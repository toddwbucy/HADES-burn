# HADES: agent onboarding

You have a command-line tool called `hades`. It is the interface to a set of
ArangoDB knowledge graphs holding source code, documents, papers, specs, and the
relationships between them. This file tells you what it is, how to approach it,
and what will go wrong if you guess.

Read the whole file before your first command. It is short, and the failure modes
are not obvious from the help text.

## What HADES is

HADES is a DBA toolkit for knowledge graphs, built for agents rather than people.
Every invocation is one self-contained `hades ...` command that prints JSON to
stdout and logs to stderr. There is no REPL, no session, no server you connect
to. If you can run a shell command, you have the whole interface.

A HADES database is not a document store you full-text search. It is a typed
graph: named document collections (files, symbols, chunks, axioms, specs), named
edge collections that are themselves the relation types, and named graphs that
group edges into a traversable whole. Meaning lives in the edges as much as the
documents.

What HADES is for, in one sentence: give a model durable, queryable memory of a
codebase or corpus that outlives any one context window.

## Rules you do not break

1. **Production data is read-only.** ArangoDB ACLs on the `hades` user are the
   real gate, and production research databases are granted read-only. Do not
   attempt writes against them. Reading any database is fine. If you need to
   write, use `bident_burn` or create your own test database.

2. **Never drop a database.** There is deliberately no `db drop-database`
   command. Dropping is a human act performed in the ArangoDB console. Do not
   look for a workaround.

3. **Do not hand-write AQL for writes.** `db aql` exists and is read-only.
   Mutating AQL is rejected. Use the structured `db` operations for writes. The
   restriction is the guardrail, not an inconvenience to route around.

4. **Always name the database.** `hades --db <name> <command>`. There is no
   implicit default. A command without `--db` is a bug waiting to hit the wrong
   store.

5. **Know what stderr is for before you silence it.** Output is JSON on stdout,
   logs and errors on stderr. They are separate streams, so a pipe to `jq` gets
   clean JSON already:

   ```bash
   set -o pipefail
   hades --db NL db collections | jq '.data'
   ```

   `2>/dev/null` is **not** required for parsing, and it is not free. On failure
   `hades` writes the diagnostic to stderr and exits non-zero, so suppressing
   stderr turns a connection, permission, or validation failure into silent
   empty output with nothing to explain it. Suppress only when your harness
   merges the two streams and log lines would corrupt the JSON, and check the
   exit code when you do.

   Use `set -o pipefail`. Without it a pipeline reports the **last** command's
   status, so a failed `hades` piped into anything exits 0 and the failure
   vanishes.

   When something behaves unexpectedly, drop the suppression first: the answer
   is usually already on stderr.

## Your first five minutes

Do these in order. Skipping step 2 is the single most common way agents conclude
"search is broken" when it is not.

```bash
# 1. What databases can I see? (--db is required even here: the listing is
#    scoped to the connection, so name any database you can already reach)
hades --db <any-reachable-db> db databases

# 2. What is actually in this one, and which profile holds it?
hades --db <name> orient
hades --db <name> db stats

# 3. What collections and graphs exist?
hades --db <name> db collections
hades --db <name> db schema list

# 4. Now search, naming the profile you learned in step 2.
hades --db <name> db query "your question" -c codebase -n 10
```

On step 1: the listing endpoint is scoped to whatever `--db` names and returns the
databases the connecting user can access, so every reachable database returns the
same list. `_system` is the conventional guess when you have no name at all, but
access is granted per database, so `_system` may be the one database this user
cannot reach. A 401 or 403 there is a permissions answer, not a broken install.

## The one concept that trips everyone: profiles

`db query` does not search "the database". It searches a **collection profile**,
which is a triple of (metadata, chunks, embeddings) collections. Common profiles
are `default` and `codebase`.

With no `-c`, you get the `default` profile, which in many databases is empty.
The exception is worth knowing because it is invisible: if
`HADES_DEFAULT_COLLECTION` is set in the environment, it names the default
instead, and nothing in the output tells you. Pass `-c` explicitly when it
matters.

If a search against a clearly populated database returns nothing, or errors with
`404 collection embeddings`, you have the wrong profile. You do not have a broken
index. Run `db stats` first: it prints per-profile totals, so it tells you which
profile has data before you spend a query.

```bash
hades --db weavertools db stats            # which profile is populated
hades --db weavertools db query "axiom" -c codebase -n 10
```

Flags on `db query`: `-n` result count, `-H` hybrid (vector plus keyword), `-S`
structural (uses trained graph embeddings). `-R`/`--rerank` parses but exits
non-zero, because the cross-encoder it needs does not ship with the CLI. Use `-H`
or `-S` instead.

## Command map

| Group | Commands |
|---|---|
| Orientation | `status`, `orient`, `db stats`, `db collections`, `db databases`, `db health` |
| Search | `db query "text" -c <profile> -n 10 [-H] [-S]`, `db aql '<read-only>' -b '{...}'` |
| Read | `db get <col> <key>`, `db list -c <profile>`, `db count <col>`, `db export <col>`, `db check <id>` |
| Write | `db insert <col> --data '[...]'`, `db update <col> <key> --data '{...}'`, `db delete <col> <key> -y` |
| Collections | `db create <name> -t document\|edge`, `db truncate <col> -y`, `db drop-collection <col> -y`, `db create-index`, `db index-status` |
| Graph | `db graph list`, `db graph neighbors <id> --graph <g>`, `db graph traverse <id> --graph <g>`, `db graph shortest-path`, `db graph create`, `db graph materialize` |
| Schema | `schema apply <file.yaml> [--dry-run] [-y]`, `db schema {init,list,version}`, `db schema show <name>` |
| Code graph | `codebase ingest <path>`, `codebase drift <path>`, `codebase validate`, `codebase stats`, `codebase prune-orphans`, `codebase retire`, `codebase update` |
| Documents | `ingest <files...>`, `extract <file>` |
| Embeddings | `embed text "..."`, `embed service ...`, `graph-embed train`, `graph-embed update` |

Run `hades <command> --help` before composing anything non-trivial. It is the
best available source and cheaper than a wrong guess, though it describes what a
flag is *for* rather than whether it currently works: `-R` above is the example.

Note on short flags: `-g` is the global `--gpu`. It is **not** an alias for
`--graph`. Graph commands take the long form, `--graph <name>`.

## Common jobs

### Understand an unfamiliar codebase that is already ingested

```bash
hades --db <name> db stats
hades --db <name> db query "how does authentication work" -c codebase -n 10
hades --db <name> db graph neighbors "codebase_files/src_auth_rs" --graph codebase_graph
```

Search finds entry points. Traversal tells you what they connect to. Use both:
the graph is the part a plain grep cannot give you.

### Ingest a codebase

```bash
hades --db <name> codebase ingest /path/to/repo
```

Ingest at the **repository root, as a directory**, not file by file. A node key is
its path relative to the ingest root with `.` and `/` replaced by `_`. Ingesting
piecemeal produces inconsistent keys and edges that do not resolve.

If the tree contains any `.rs` or `.go` file, rust-analyzer and gopls are probed
before the first file is written, and a tool that will not run aborts the entire
run with exit 1 and nothing ingested. This is not specific to `--force`, and it
is the first failure most agents meet. See the analyzer notes under `changed`
below for what to do about it.

Analysis degrades in tiers rather than failing. A recognized extension is sent to
its semantic analyzer first; if that is unavailable or errors, the file falls
back to a Tree-sitter parse; only if Tree-sitter also cannot recover structure
does the file land on the raw-text path. The Tree-sitter grammars are compiled
into the binary, so that last step is rare and means the source itself is
malformed, not that a tool is missing on the host.

Files with an unhandled extension are skipped. Extensionless files are classified
by shebang, so a script starting with `#!/bin/bash` is picked up and stored as raw
text. Use `--unparsed-ext wgsl,vert` to bring extra extensions in through the
raw-text path. The flag only applies to extensions no analyzer already claims, so
naming one that does (`cu` and `cuh` are already C++) changes nothing.

Which tier a file landed on is recorded on its node as `analysis_tier`
(`semantic`, `structural`, `text`). Extension is not a proxy for it, and the tier
determines how change detection behaves on that file.

### Check whether a graph still matches its source tree

```bash
hades --db <name> codebase drift /path/to/repo
```

This is the honest health check. The JSON reports:

- `stale.count`: graph nodes whose source file no longer exists
- `uningested.count`: source files with no node
- `changed.count`: matched files whose content differs from what was ingested
- `changed.unverifiable`: matched files that could **not** be compared, because
  they were ingested before HADES recorded a content hash, or are no longer
  readable as text. Note it is nested under `changed`, not top level.
- `unhandled.count`: files under the root ingest has no handler for, each with a
  `reason`
- `clean`: true only when `stale`, `uningested`, `changed` **and**
  `changed.unverifiable` are all zero

`unverifiable` blocks `clean` on purpose. A file nobody compared is not a file
known to be current, and reporting a clean sweep over uncompared files is exactly
the false green this command exists to remove.

Any graph built before `content_hash` existed reports `clean: false` with a large
`unverifiable`. The remedy is `codebase ingest --force <the original ingest
root>`. Both halves matter. Do not narrow the path to bound the cost: keys are
derived relative to whatever path you pass, so a subdirectory re-bases every key
under it, the `unverifiable` count does not drop, and the subtree is duplicated.
The `--force` is load-bearing too: a plain re-ingest skips every file whose
`symbol_hash` still matches and returns before it writes the file document that
carries `content_hash`, so the count does not move. The exception is a file
counted as unverifiable because it no longer reads as text. Re-ingest cannot read
it either, so that one stays counted until the file is readable again or its node
is retired.

`unhandled` does **not** block `clean`, because every repository contains a
README. Read the bucket and judge for yourself whether the gap matters.

`changed` is worth understanding. Drift always compares a full-content hash, but
incremental re-ingest compares the file node's `symbol_hash`, and what that field
covers depends on which analyzer produced it. That mismatch is the whole reason
the bucket exists.

- **Python and Rust at tier `semantic`**: `symbol_hash` covers symbol **names**
  only. An edit to a body, a signature, or a comment leaves it identical, so a
  plain `codebase ingest` skips the file while drift keeps reporting it. This is
  the case that needs `--force`.
- **Tier `structural` (any language), and C++ at tier `semantic`**: the digest
  covers the serialized symbol list, including line spans and metadata. Most
  edits move a line and so change it, and a plain re-ingest picks the file up.
  An edit that leaves every symbol boundary and every recorded attribute
  identical will still be missed.
- **Tier `text`**: `symbol_hash` is the full-content digest, so any edit at all
  changes it and a plain re-ingest always picks the file up.
Go sits in the second bucket: it has no per-file semantic analyzer, so it is
Tree-sitter-analyzed and carries the serialized digest. The gopls phase adds
semantic symbols and edges afterwards, and those are recorded on the node under
their own keys (`gopls_analyzed`, `gopls_symbol_count`) rather than by restating
the file's tier.

Read the tier off the node rather than guessing from the extension. `--force` is
the broadest remedy: it rebuilds symbols, chunks and embeddings for every file
under the ingest root.

`--force` never permits an analyzer-fidelity downgrade on its own, and that shows
up in two very different ways.

**Analyzer missing: the whole run refuses, up front.** rust-analyzer and gopls
are probed before any file is touched. If the tree contains `.rs` or `.go` files
and the tool is not runnable, ingest exits non-zero with
`<name> preflight failed (...)` having written nothing. It stops early on
purpose, because `--force` purges semantic edges the analyzer would rebuild and
aborting after the purge would be too late. So on a host without rust-analyzer,
`codebase ingest --force /repo` to clear a large `unverifiable` moves no counter
at all. `--allow-analysis-downgrade` is not an afterthought here, it is the only
thing that lets the command run.

**Stored analysis outranks incoming: that file is skipped, and `--force` does not
override it.** When a lower-tier analysis would permanently replace a richer
stored one, the file is returned as skipped with
`error: "higher-fidelity stored analysis preserved"` and drift keeps reporting
it. This guard runs *before* the `--force` check and is gated on
`--allow-analysis-downgrade` instead.

The case to expect is a semantic analyzer that was available at first ingest and
is not now: C++ ingested without the compilation database it had before, so
libclang loses to the stored `semantic`. Rust and Go do not hit this, because
their semantic artifacts come from a phase that re-runs later in the same ingest,
and the guard yields when that is scheduled. `--allow-analysis-downgrade` is the
way through when you do want the poorer analysis stored.

### Verify graph integrity

```bash
hades --db <name> codebase validate
```

Checks internal invariants (edge endpoints resolve, chunk-to-file references are
sound, keys are deterministic). Note that `validate` and `drift` answer different
questions. `validate` asks "is the graph self-consistent". `drift` asks "does the
graph still describe the tree". A graph can pass one and fail the other.

If `validate` reports dangling edges after a forced re-ingest, that is expected:
rebuilding a file can drop a symbol another file points at. HADES leaves those
edges in place rather than deleting them, because each one records a real
dependency.

The two commands name this differently, which matters if you grep. `codebase
ingest` reports the count in its own output under `dangling_inbound_edges`.
`codebase validate` has no such key: it emits one record per invariant, shaped
`{id, name, description, violations, violation_count}`, and unresolved endpoints
appear under the invariant **names** `defines_edge_endpoints`,
`calls_edge_endpoints`, `implements_edge_endpoints` and
`imports_edge_endpoints`. Searching validate output for `dangling_inbound_edges`
finds nothing and reads as a clean graph.

Two repairs, in order of preference. Re-ingesting re-resolves the relation and is
non-destructive:

```bash
hades --db <name> codebase ingest --force /path/to/repo   # the original root
```

Both parts of that command are load-bearing, and each is easy to get wrong.

`--force` is required. The dependent file is dependent precisely because the
*other* file changed, so its own symbol set is untouched, its `symbol_hash` still
matches, and a plain `codebase ingest` skips it. A skipped file is never
**re-purged**, and the purge is what removes edges pointing at symbols that no
longer exist. The rust-analyzer and gopls phases do still run over skipped files,
because their file lists are built during discovery, but they only add fresh
edges by deterministic key. Nothing deletes the dead ones. A plain re-ingest over
an otherwise-unchanged tree leaves every dangling edge exactly where it was, with
a correct edge sitting beside each one.

**The path must be the original ingest root**, even though only a few files need
repairing. Keys are derived relative to whatever path you pass: a directory bases
at itself, and a *file* bases at its parent. So re-ingesting one file writes a
node keyed `main_rs` instead of `crates_hades-cli_src_main_rs`. That purges
nothing, leaves the dangling edges in place, and adds a duplicate node that drift
then reports as one extra `stale` key. Note what it does **not** do: the real
file's node is still there and still matches the tree, so it stays `matched` and
`uningested` stays empty. A zero there is not confirmation the repair worked.
This is the same rule as ingesting
at the repository root, and it means the narrow-looking repair is in fact a
whole-tree rebuild. Budget for it.

The second repair, `codebase prune-orphans`, drops the edges instead. It is
destructive, so preview it first:

```bash
hades --db <name> codebase prune-orphans --dry-run   # a floor, not the set
# read the report and its notes, then:
hades --db <name> codebase prune-orphans
```

Read `--dry-run` as a **lower bound, not the deletion set**. The sweep is
cascade-ordered: a real run deletes orphan symbols first, which strands the edges
pointing at them, and those get deleted too. A dry-run deletes nothing, so those
edges still resolve and are not counted. The tool says so itself, in
`dangling_edges_note` and `orphan_embeddings_note` on the dry-run JSON. Expect a
real run to remove at least what the preview showed, and possibly more. Deleting
an edge is deleting a recorded relationship, so never run it unreviewed.

### Build a new graph database

The flow is create, declare schema, ingest, verify.

```bash
set -o pipefail
hades --db <any-reachable-db> db create-database mygraph
hades --db mygraph schema apply schema.yaml --dry-run   # review the plan
hades --db mygraph schema apply schema.yaml             # no -y on a new database
hades --db mygraph codebase ingest /path/to/repo
hades --db mygraph codebase validate
```

`db create-database` overrides whatever `--db` names and connects to `_system`
internally, since that is the only context ArangoDB accepts database creation
from. Passing `--db _system` yourself does nothing. What the command does need is
a `_system` grant on the connecting user, and no `--db` value substitutes for
one: a 401 or 403 here is a permissions answer, not a wrong flag.

The schema YAML declares `collections`, `edge_definitions`, `named_graphs`, and
optionally `relation_order` plus `feature_dim` and `model_type` for structural
training. Always `--dry-run` first. Validation rejects duplicate collections,
edge endpoints that are not document collections, and unknown model types.

**Do not habitually pass `-y` to `schema apply`.** `-y` is `--force`. It skips
the in-use guard, which is the check that refuses to apply a schema to a database
that already holds data in the declared collections, and existing documents are
then **overwritten by `_key`**. On a genuinely new database the guard never
trips, so `-y` buys nothing and only trains the habit. If an apply is refused,
that refusal is information: read it, confirm the overwrite is what you intend,
and only then re-run with `-y`.

## Reading failures correctly

| Symptom | What it usually means |
|---|---|
| Silent empty output | The command failed and you suppressed stderr. Re-run without `2>/dev/null` and check the exit code |
| A pipeline "succeeds" but produces nothing | No `set -o pipefail`, so a failed `hades` exited 1 and the pipeline reported the last command's 0 |
| JSON parse error | Your harness merged stderr into stdout. Not caused by omitting `2>/dev/null`, which never affects a pipe. Omitting `--db` cannot produce this: it errors on stderr and leaves stdout empty, which is row 1 |
| Search returns nothing on a populated DB | Wrong profile. Run `db stats` and pass `-c` |
| `404 collection embeddings` | Same. Wrong profile, not a broken index |
| Drift reports near-total drift both ways | Wrong ingest root. Keys are relative to it |
| `invalid value for '--gpu'` | You used `-g` expecting `--graph`. Use the long form |
| Mutating AQL rejected | Working as designed. Use structured `db` operations |

## How to work with this tool

Prefer the narrowest command that answers your question. `orient` and `db stats`
are cheap and stop you guessing. Traversal beats repeated search when you already
have a node id. Read `--help` for a command before composing a long invocation.

When a command reports something you did not expect, read the report rather than
working around it. This toolkit is deliberately built to surface gaps instead of
smoothing over them, so a bucket you did not expect is usually the tool telling
you something true about the graph.
