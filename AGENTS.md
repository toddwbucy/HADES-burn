# Repository Guidelines

## Project Structure & Module Organization

HADES-Burn is a Rust 2024 workspace. `crates/hades-cli` provides the `hades` binary and command dispatch; `crates/hades-core` contains database, graph, pipeline, and code-analysis logic; `crates/hades-proto` builds gRPC types from `proto/`; and `crates/hades-prefetch` supports graph-aware training. Rust unit tests live beside code, while integration tests live under each crate's `tests/` directory. GPU-bound Python services and their pytest suite are in `services/`. Keep deployment files in `services/systemd`, defaults in `config/`, documentation in `docs/` or `Bastion/`, and installation utilities in `scripts/`.

## Build, Test, and Development Commands

- `cargo build --workspace` builds the debug CLI and all libraries; `cargo build --release` produces `target/release/hades`.
- `cargo fmt --all -- --check` verifies Rust formatting.
- `cargo clippy --workspace --all-targets -- -D warnings` runs the same strict lint policy as CI.
- `cargo test --workspace --lib` runs the CI unit-test set.
- `cargo test --workspace` also runs integration tests; machine-dependent tests self-skip when services are unavailable.
- `cd services && make test` runs Python tests with pytest. `make proto-gen` regenerates Python protobuf stubs.

`protoc` is required when building `hades-proto`.

## Coding Style & Naming Conventions

Accept `rustfmt` defaults and use four-space indentation in Python. Follow idiomatic naming: `snake_case` for modules, functions, and tests; `CamelCase` for Rust types; and `SCREAMING_SNAKE_CASE` for constants. Declare shared Rust dependencies in the root `Cargo.toml`. Use `thiserror` in libraries, `anyhow` at CLI boundaries, and `tracing` for diagnostics. CLI results belong as JSON on stdout; logs and progress belong on stderr.

## Testing Guidelines

Name Rust integration files after the behavior or subsystem and Python tests `test_*.py`. Add focused tests with every behavior change; no numeric coverage threshold is defined. External-resource Rust tests must document prerequisites, skip clearly when unavailable, and honor `ARANGO_TESTS=1` as strict mode. See `docs/specs/workstation-specific-tests.md`.

## Commit & Pull Request Guidelines

History follows Conventional Commit-style subjects such as `feat(training): ...`, `fix(graph): ...`, and `docs(bastion): ...`. Keep commits scoped and imperative; append issue or PR references when relevant. Pull requests should explain motivation and user-visible effects, link tracking issues, list verification commands, and update protocol/configuration docs when contracts change.

## Security & Configuration

Never place credentials in YAML or commits; use environment files or variables. Production ArangoDB data is sacrosanct: run write tests only against `bident_burn` or a dedicated test database, and rely on ArangoDB ACLs as the authoritative write boundary.
