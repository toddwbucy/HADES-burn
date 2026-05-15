# Workstation-specific tests

**Status:** accepted, documents existing practice
**Issue:** [#93](https://github.com/toddwbucy/HADES-burn/issues/93)
**Date:** 2026-05-15

## Problem

Some tests in this repository depend on resources that exist only on a
specific machine: a running ArangoDB instance with particular data, a
locally-available embedder service, a real `NestedLearning` database
with populated collections. These tests cannot generalize — renaming
identifiers doesn't help, because the tests *do* what their names
say *for that specific data*.

The risk is silent drift. If a test silently skips when its environment
is unavailable, CI green tells us nothing about whether the test
actually exercises anything. Over time the test ages out, the
environment shifts, and no one notices until someone needs the test to
pass and discovers it never has.

## Decision

Workstation-specific tests are first-class. They live in the repo, they
self-skip in environments where they can't run, and they support an
opt-in strict mode that converts skip into panic so a workstation can
verify regressions deliberately.

### Convention

A test that needs an external resource (database socket, embedder
service, specific data) follows this shape:

1. **Lives in `crates/<crate>/tests/<name>.rs`** — Rust's integration
   test convention. Such tests are not compiled when `cargo test --lib`
   runs, which is what CI runs by default. They only enter the build
   when `cargo test` (no flag) or `cargo test --test <name>` runs them
   explicitly.

2. **Skips gracefully when its environment is absent.** Use a setup
   helper that returns `Option<Setup>` and have each test do
   `let Some(setup) = setup_helper() else { return };`. Log a `warn!`
   line explaining what was missing so a contributor running tests
   locally can see the skip.

3. **Honors `ARANGO_TESTS=1` (or analogous flag) as strict mode.** When
   the strict flag is set and the environment is *still* missing, the
   setup helper panics with a message naming what's needed. This lets
   a workstation operator deliberately run "everything should work
   right now" and get a hard failure if it doesn't.

4. **Documents required environment at the top of the file.** A short
   doc comment listing the external resources the test depends on and
   the env vars it honors. Example: `tests/graph_loader.rs` lists
   socket path, password env var, and the expected database state.

### CI behavior

CI runs `cargo test --workspace --lib` by default. This compiles only
unit tests inside each crate's `src/` tree; integration tests in
`tests/` are not built, not run, and don't contribute to CI time or
flake risk.

A future workstation-class CI job could run `ARANGO_TESTS=1 cargo test
--workspace` against a runner that has the required services. Such a
job is out of scope for this spec; the current convention leaves the
door open for it.

### What this is not

- **Not a feature flag.** Cargo features were considered and rejected:
  they require remembering an extra flag, they make `cargo test`
  semantics non-obvious, and the integration-test directory already
  provides the isolation we need.
- **Not a separate repository.** Out-of-tree harness was considered
  and rejected: tests are meaningfully tied to the crates they
  exercise, and moving them across a repo boundary adds friction
  without proportionate benefit at this scale.

## Audit (2026-05-15)

All ArangoDB-dependent integration tests follow the convention:

- `crates/hades-core/tests/graph_loader.rs` — skip+strict
- `crates/hades-core/tests/arango_cache.rs` — skip+strict
- `crates/hades-core/tests/arango_crud.rs` — skip+strict
- `crates/hades-core/tests/arango_index.rs` — skip+strict
- `crates/hades-core/tests/arango_query.rs` — skip+strict
- `crates/hades-core/tests/arango_transport.rs` — skip+strict
  (brought into line with this spec; previously skip-only)

The remaining integration test files
(`config_integration.rs`, `training_client.rs`, `extraction_client.rs`,
`embedding_client.rs`, `pipeline.rs`, `proto_types.rs`) are
self-contained or mock-driven and don't depend on workstation state.

## How to add a new workstation-specific test

```rust
//! Integration tests for <thing>.
//!
//! Prerequisites:
//! - ArangoDB running with socket at /run/arangodb3/arangodb.sock
//! - ARANGO_PASSWORD environment variable set
//!
//! Tests are skipped gracefully when prerequisites are missing.
//! Set ARANGO_TESTS=1 to make missing prerequisites a hard error.

use std::path::PathBuf;
use tracing::warn;

fn require_socket() -> Option<PathBuf> {
    let socket = PathBuf::from(
        std::env::var("ARANGO_SOCKET")
            .unwrap_or_else(|_| "/run/arangodb3/arangodb.sock".to_string()),
    );
    if !socket.exists() {
        if std::env::var("ARANGO_TESTS").is_ok_and(|v| v == "1" || v == "true") {
            panic!("ARANGO_TESTS=1 but socket missing: {}", socket.display());
        }
        warn!("skipping: ArangoDB socket not found at {}", socket.display());
        return None;
    }
    Some(socket)
}

#[tokio::test]
async fn test_thing() {
    let Some(socket) = require_socket() else { return };
    // ... test against the resource
}
```

Future contributors writing a workstation-specific test should follow
this shape so the skip behavior is uniform across the suite.
