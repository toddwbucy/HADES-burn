# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
While the version is `0.x` the API and CLI surface should be considered unstable;
breaking changes may land in any minor bump and are noted under **Changed** when
they affect operator-facing behavior.

## Convention

Every PR adds its entry to the `[Unreleased]` section in the same commit, under
one of: `Added`, `Changed`, `Deprecated`, `Removed`, `Fixed`, `Security`.
When a release ships, the `[Unreleased]` section is renamed to the new version
with the release date, and a fresh `[Unreleased]` is opened above it.

## [Unreleased]

### Added

- `hades-viewer` (`crates/hades-frontend`) — a local WebGL graph viewer for
  any HADES graph, plus a shared-reference channel between a human and an
  agent. Renders a named graph as a force-directed view with styling driven
  by attributes discovered from the data, expands neighborhoods on demand,
  and makes every view addressable (`?db=&graph=&node=`) so an agent can
  hand back a link to the exact node it means; right-click copies a
  briefing with the node's attributes, connections, and runnable `hades`
  commands. Depends on no other HADES crate — it consumes the CLI's
  JSON/jsonl output only. Binds loopback or a private LAN range, validates
  the Host header, and requires `--password` for any non-loopback
  bind. (#182)

- `scripts/install/test/` — container-based install validation harness.
  Builds a fresh Ubuntu 24.04 image with ArangoDB pre-installed, then
  runs the README install steps end-to-end. Catches packaging,
  ordering, and prerequisites issues without requiring a real VPS.
  Two real issues caught and fixed in the README this round: the
  ArangoDB GPG signing key is currently expired upstream, and the
  README's step ordering required the `hades` group before
  `systemd-sysusers` had created it. (#96)

- `AGENTS.md` — a self-contained onboarding guide for an agent that has
  never seen HADES, usable as a Codex `AGENTS.md` entry point or as a
  pasted system prompt. It covers what `--help` cannot say: that `db query`
  searches a collection profile rather than "the database", how the drift
  buckets relate to what ingest actually skips, and which remedies are safe
  to narrow. (#186)

### Changed

- **Breaking (CLI):** `-g` is no longer an alias for `--graph` on
  `db graph traverse`, `db graph shortest-path`, and `db graph neighbors`.
  It collided with the global `--gpu -g`, which made clap's uniqueness
  assertion fire on every invocation of those three commands in debug
  builds. Release builds resolved `-g` to `--graph`, so scripts using the
  short form worked there and now need the long `--graph` form; `-g` is
  the global `--gpu` everywhere. `scripts/cli_audit.sh` updated. (#182)

- README **Install** section rewritten end-to-end. Drop the WIP banner
  (the procedure has been validated via the harness), add prerequisites
  (ArangoDB, Rust toolchain, protoc), fix the step ordering so
  systemd-sysusers runs before any command that references the `hades`
  group, add explicit `mkdir -p /etc/hades` and `sudo` invocations
  where they were missing, and add a verification step at the end. (#96)
- Verified by full bident_burn re-ingest after #110 landed: coverage
  reached 99.95% (2109/2110 chunks embedded). Single outlier is a
  stale chunk record from a deleted file pre-fix; AC for #98 satisfied.

### Fixed

- Operator-facing help text that contradicted the implementation. `db query
  --rerank` advertised itself as "Enable re-ranking of results" while the
  flag exits non-zero without searching, and now says so. `codebase drift`
  still described the pre-#183 output, and now documents `changed`,
  `unhandled`, `changed.unverifiable` and `clean`, including the fact that
  what `symbol_hash` covers depends on the node's `analysis_tier`. The
  `codebase ingest --force` help and the runtime dangling-edge warning both
  told operators to "re-ingest the dependent files", which either no-ops
  (the dependents' own `symbol_hash` is unchanged, so they are skipped) or
  writes duplicate nodes under re-based keys (a narrower path re-bases every
  key beneath it); both now name `--force` and the original ingest root.
  (#186)

- `codebase drift` reported a clean sweep over partially-covered trees.
  Files with no ingest handler fell outside drift's notion of source
  entirely — neither ingested nor reportable — so `stale=0 uningested=0`
  was returned for a tree ingest had only partly read. Drift now reports
  an `unhandled` bucket with a per-file reason, and a `clean` flag that
  is false whenever anything is unhandled or changed. (#183)
- `codebase drift` could not see content staleness at all. `symbol_hash`
  is deliberately name-only (a rewritten body, changed signature, or
  edited comment leaves it identical), and drift compared only file
  existence, so an edited file reported clean while its stored chunks and
  embeddings were stale. Ingest now records a full-source `content_hash`
  alongside it and drift reports a `changed` bucket. Files ingested before
  this change are counted as `unverifiable` rather than assumed clean.
  Incremental re-ingest behavior is unchanged — `--force` still refreshes
  such a file. (#183)
- Extensionless scripts were invisible to ingest. `--unparsed-ext` is
  extension-keyed, so a file named `deploy-thing` with a `#!/bin/bash`
  first line could not be named by any flag. Discovery now sniffs the
  shebang of extensionless files: a recognized interpreter selects the
  analyzer (`#!…python3` → Python), and any other shebang routes the file
  to the raw-text path so its content is at least visible. (#183)
- `codebase ingest` gave no signal when a rebuild left inbound edges
  dangling. A rebuild that drops a symbol (rename, re-qualification,
  analyzer change) leaves `codebase_imports_edges` from *other* files
  pointing at nothing, breaking the `imports_edge_endpoints` invariant
  with nothing in the output to say so. Ingest now reports
  `dangling_inbound_edges` in its JSON summary, scoped to the files it
  rebuilt and to targets that genuinely do not resolve, and `--force`
  documents the repair. They are reported rather than deleted: each edge
  records a real dependency, and removing it would erase the only signal
  that the dependent needs re-ingesting — the dependent is unchanged, so
  every later ingest skips it and never re-derives the relation. Re-ingest
  the dependents, or run `codebase prune-orphans` to drop them. (#183)

- Embedding coverage gap during `codebase ingest`. Two root causes:
  the embedder service OOMed on per-file batches whose padded sequence
  lengths exceeded available GPU memory (typically on large source
  files like `dispatch.rs` with 90+ chunks of variable token-length),
  and the ingest code swallowed the resulting HTTP 500 into a
  `warn!` log line that never reached the user-facing JSON output —
  so files appeared ingested successfully while their embeddings
  were silently dropped. Two coordinated fixes:
  - `EmbeddingClient::embed` now splits requests at `batch_size`
    and progressively halves on OOM-shaped errors, recursively down
    to single chunks. Results reassembled by start index regardless
    of completion order.
  - `codebase ingest` surfaces per-file embedding failures via a new
    `embedding_error` field on each file's JSON result, plus a
    `files_with_embedding_failures` count and `embedding_failure_paths`
    list in the top-level summary. Failures are no longer silent.
  Verified on `dispatch.rs` (97 chunks, was 0/97 → now 96/96) and
  on a full re-ingest of `bident_burn`. (#98)

### Added

- Spec doc `docs/specs/workstation-specific-tests.md` codifying the
  skip-or-strict pattern for integration tests that depend on
  workstation-specific resources (live ArangoDB, embedder service,
  specific database state). Convention is the existing practice in
  `tests/graph_loader.rs` and the `arango_*` integration tests: live
  in `tests/`, skip when prerequisites are absent, panic when
  `ARANGO_TESTS=1` is set and prerequisites are still absent. (#93)
- `crates/hades-core/tests/arango_transport.rs` brought in line with
  the spec: previously skipped without honoring `ARANGO_TESTS=1`
  strict mode; now uses the shared `require_socket()` helper that
  panics under strict mode. (#93)

### Changed

- Strip arxiv- and NestedLearning-specific defaults from test fixtures,
  comments, and operator-surface documentation. `NestedLearning` is no longer
  used as a sample database name in tests; `arxiv_metadata` is no longer used
  as a sample collection name. `CLAUDE.md`'s "Production data is sacrosanct"
  paragraph updated to reflect the ArangoDB-ACL-based security model rather
  than the (since-removed) compile-time allowlist. Obsolete write-guard
  assertion removed from `scripts/cli_audit.sh`. (#92)
- README: replace the "Three-tier access control via SO_PEERCRED" claim with
  an accurate description of what the tier dispatch actually does — opt-in
  client self-restriction, useful as a UX guard for AI-agent harnesses.
  Security against malicious clients is enforced at the ArangoDB layer via
  ACL grants on the `hades` user; the daemon's tier dispatch is not a
  security boundary. (#97)

### Added

- `CHANGELOG.md` (this file), with the convention documented above. (#95)

## [0.3.0] - 2026-05-14

This is the baseline release: the state of `main` at the point CHANGELOG
discipline began. Entries are reconstructed from PR history.

### Added

- Python call-graph extraction via AST. `codebase ingest` of Python source
  now populates `codebase_calls_edges` using a rustpython-parser AST walk
  plus a three-strategy resolver (exact qualified-name match, `self.method`
  → `ParentClass.method` rewrite, bare-name fallback). Parallel to the
  existing rust-analyzer-driven path for Rust. (#91)
- `.gitignore` and `.hadesignore` are honored during codebase ingestion.
  Replaces the hand-curated `SKIP_DIRS` const with `ignore::WalkBuilder`,
  which respects standard ignore files plus a custom HADES-specific
  filename for exclusions that don't belong in version control. The hard
  `SKIP_DIRS` floor (`__pycache__`, `node_modules`, `target`, `venv`,
  `dist`, `build`) is preserved for repos with no ignore files. (#90)
- Code smell `_key` convention and `hades_burn_self` self-analysis
  database for dogfooding-driven schema work. (#86)

### Changed

- ArangoDB ACLs replace the compile-time `WRITABLE_DATABASES` allowlist
  for write-safety. HADES now connects as a dedicated `hades` ArangoDB
  user; write restrictions on specific databases are enforced by
  ArangoDB grants on that user, not by a Rust const checking the
  database name. Operator manages per-database access in arangosh.
  Removed 23 in-process call sites of `require_writable_database()`,
  the `WriteDenied` error variant, and 8 tests that exercised the
  removed guard. README install section documents the new bootstrap
  flow. (#89)
- Multiple README revisions clarifying the research-program context,
  the Persephone Embedding API contract, and the schema-as-data
  architectural commitment. (no PR — direct commits)

[Unreleased]: https://github.com/toddwbucy/HADES-burn/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/toddwbucy/HADES-burn/releases/tag/v0.3.0
