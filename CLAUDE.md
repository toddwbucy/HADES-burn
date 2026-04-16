# CLAUDE.md — HADES-Burn

## Project

Rust rewrite of the HADES knowledge graph system. Refactor of the production Python system at `~/git/HADES/`.

## Build

```bash
cargo build              # debug build
cargo build --release    # release build
cargo test               # run all tests
cargo clippy             # lint
```

Binary: `target/debug/hades` (or `target/release/hades`)

## Workspace Crates

- **hades-cli** — Binary entry point (`hades`). CLI parsing (clap), dispatch.
- **hades-core** — Library. Config, ArangoDB client, graph engine, pipeline logic.
- **hades-proto** — Library. gRPC/protobuf definitions for Persephone provider protocol.
- **hades-prefetch** — Library. Async graph-aware batch prefetcher for training.

## Critical Rules

1. **Production data is sacrosanct.** Never write to NestedLearning or any production ArangoDB database. Only `bident_burn` is writable for project management. All write-tests use `bident_burn` or a dedicated test database.
2. **Use the HADES CLI** (`hades --db bident_burn`) for all project management tasks. Dog-food the tool we're rewriting.
3. **ArangoDB socket:** `/run/arangodb3/arangodb.sock` — the existing socks proxy. Do not create new socket infrastructure.
4. **Embedder service:** `/run/hades/embedder.sock` — the existing Jina V4 service. Do not modify.

## Conventions

- Edition 2024, resolver 2
- Workspace dependencies in root `Cargo.toml`
- `thiserror` for library errors, `anyhow` for binary/CLI errors
- `tracing` for structured logging (not `log` or `println!`)
- All CLI output is JSON to stdout, progress/logs to stderr
