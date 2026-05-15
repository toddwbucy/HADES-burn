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

### Fixed

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
