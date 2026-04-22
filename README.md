# HADES-Burn

**High-speed ArangoDB Data Embedding System — Rust Implementation**

HADES-Burn is research infrastructure for low-latency retrieval-augmented generation over persistent, ontology-grounded knowledge graphs. It was developed as the context-management substrate for the Nested Learning research program — a multi-year effort to implement and validate published work on nested-optimization memory systems from the Mirrokni et al. research group (arXiv:2512.24695, arXiv:2501.00663) — and has been empirically validated through daily use as the retrieval backend for both local inference (24–32B Qwen and Mistral variants) and commercial coding agents (Claude Code) during the project's own development.

The design premise under investigation is that retrieval latency, not model capability, is frequently the binding constraint on sustained agent coherence — and that a Rust daemon connected to ArangoDB over Unix sockets, paired with an ontology-grounded schema and a closed operation vocabulary, can reduce that constraint to a level at which local, on-premises agent workflows become viable.

For the research questions HADES-Burn was built to investigate, see [RESEARCH_GOALS.md](RESEARCH_GOALS.md).

## Architecture

```
┌──────────────────────────────────────────────────────┐
│                  hades (CLI)                          │
│          14 command groups, ~85 subcommands          │
└──────────────┬───────────────────────┬───────────────┘
               │                       │
   ┌───────────▼──────────┐  ┌─────────▼────────────┐
   │     hades-core        │  │     hades-proto       │
   │  ArangoDB client,     │  │  gRPC/protobuf for    │
   │  graph engine,        │  │  Persephone provider  │
   │  code analysis,       │  │  protocol             │
   │  pipeline, chunking   │  │                       │
   └───────────┬───────────┘  └───────────────────────┘
               │
   ┌───────────▼──────────┐
   │   hades-prefetch      │
   │  Async graph-aware    │
   │  batch prefetcher     │
   │  for GNN training     │
   └───────────────────────┘
```

**External services** (GPU-bound, Python):
- Embedder — Jina V4 (Qwen2.5-VL-3B + LoRA) via gRPC on `/run/hades/embedder.sock`
- Extractor — Docling VLM via gRPC on `/run/hades/extractor.sock`

**Database**: ArangoDB over Unix socket at `/run/arangodb3/arangodb.sock`.

## Design Decisions Relevant to the Research Questions

**Unix sockets end-to-end.** Every transport in the system — client to daemon, daemon to database, daemon to embedder, daemon to extractor — uses a Unix domain socket. There is no HTTP or TCP path in the query hot path. This is the single most consequential latency decision and sets the floor for what the rest of the system can achieve.

**Closed operation vocabulary for model agents.** Models do not write raw AQL against HADES-Burn. The daemon exposes a bounded set of pseudo-code operations (`search`, `traverse`, `neighbors`, `materialize`, and so on) that are translated to AQL internally. The vocabulary is deliberately aligned with the training distribution of 24–32B-parameter Mistral and Qwen models, so that operation selection is reliable without task-specific fine-tuning. This also functions as a guardrail: the action space available to a model agent is finite and inspectable. See [docs/model-operation-vocabulary.md](docs/model-operation-vocabulary.md).

**Ontology as data, not code.** Database ontologies are stored in the `hades_schema` collection and loaded at runtime. Schema evolution is a database operation, not a compile-and-deploy cycle. This makes it practical to run multiple research configurations from a single binary.

**Three-tier access control via SO_PEERCRED.** The query daemon at `/run/hades/hades.sock` uses peer-credential-based sessions with three access tiers — Agent (safe reads and task management), Internal (diagnostics), and Admin (writes, DDL, raw AQL). Tier assignment is a function of Unix peer credentials, not a token exchange. See [docs/daemon-protocol.md](docs/daemon-protocol.md).

**AST-level code ingestion.** Rust source is parsed with `syn` and rust-analyzer; Python with `rustpython-parser`. Chunking respects symbol boundaries, and cross-file import resolution produces typed edges in the graph. Embedding is optional and decoupled from structural ingestion.

## Requirements

**Runtime:**
- [ArangoDB Community Edition](https://arango.ai/downloads/) — the persistent graph store. Required for all database-backed commands.

For low-latency local deployments, HADES-Burn connects to ArangoDB over a Unix domain socket. If you are running local models and want the low-latency transport path the system was designed around, the [`arango-unix-proxy`](https://github.com/r3d91ll/arango-unix-proxy) project provides the socket proxy configuration.

**Build:**
- Rust edition 2024 (stable 1.85+)

**Optional — only required for their respective command paths:**
- Jina V4 embedder service on a Unix socket (default `/run/hades/embedder.sock`) — required for `hades embed`, `hades codebase ingest --embed`, and hybrid-search queries.
- Docling extractor service on a Unix socket (default `/run/hades/extractor.sock`) — required for `hades extract` (paper / PDF ingestion).

## Build

```bash
cargo build              # debug
cargo build --release    # release
cargo test               # ~350 tests
cargo clippy             # lint
```

Binary: `target/debug/hades` or `target/release/hades`.

## Usage

```bash
hades --db <name> db query "attention mechanism" -n 5
hades --db <name> db graph traverse "papers/arxiv_2501_00663" -d outbound --max-depth 3
hades --db <name> codebase ingest /path/to/project --lang rust
hades --db <name> db schema init --seed nl
hades daemon
```

All output is JSON to stdout; progress and logs are written to stderr. The full command set is documented in [docs/daemon-protocol.md](docs/daemon-protocol.md).

## Project Structure

```
crates/
  hades-cli/        # Binary entry point; clap parsing, command dispatch
  hades-core/       # Config, ArangoDB client, graph engine, pipeline, chunking
  hades-proto/      # gRPC/protobuf for Persephone provider protocol
  hades-prefetch/   # Async graph-aware batch prefetcher for GNN training
```

## Project Status

As of v0.3.0 — production cutover from the prior Python implementation is complete; all CLI commands are native Rust.

| Metric | Value |
|--------|-------|
| Rust source files | 100 |
| Lines of Rust | ~36,000 |
| Tests | ~350 |
| PRs merged | 66 |
| CLI subcommands | ~85 (all native Rust) |

Not yet ported: `--rerank` (cross-encoder model, deferred until ONNX runtime integration).

## Authorship and Development Process

HADES-Burn was designed, directed, and validated through daily use by the author, with implementation assistance from Claude Code and other contemporary AI coding agents. Research direction, architectural decisions (Unix-socket transport layer, closed model-operation vocabulary, ontology-grounded schema, three-tier access model, data-sacrosanct operational boundaries), integration with the broader Nested Learning research program, and the dogfooding methodology are the author's work.

The project serves simultaneously as research infrastructure and as a long-running empirical study of AI-assisted development workflows. The commit and refactor history is itself data on what retrieval and context primitives make AI coding agents effective at sustained engineering work — a research question HADES-Burn is positioned to study precisely because it was built under those conditions.

## Contributing

The project is open to collaboration from academic and industry researchers working on retrieval latency, local LLM deployment, agent memory systems, or ontology-grounded retrieval. Issues and pull requests are welcome.

## Documentation

| Document | Description |
|----------|-------------|
| [RESEARCH_GOALS.md](RESEARCH_GOALS.md) | Research questions, context-management framing, measurement points |
| [docs/daemon-protocol.md](docs/daemon-protocol.md) | Wire protocol, session model, access tiers, command reference |
| [docs/model-operation-vocabulary.md](docs/model-operation-vocabulary.md) | Closed operation set for AI model agents |
| [docs/codebase-graph-ontology.md](docs/codebase-graph-ontology.md) | Universal code ontology — collections, edges, named graph |
| [docs/og-rag-report.md](docs/og-rag-report.md) | Ontology-grounded hypergraph retrieval analysis and integration proposal |
| [docs/design-agent-memory-and-system-prompt.md](docs/design-agent-memory-and-system-prompt.md) | Cross-project design for agent memory and system prompt bootstrap |

## License

Licensed under the [Apache License, Version 2.0](LICENSE). See `LICENSE` for the full text.
