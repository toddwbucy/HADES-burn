# HADES-Burn

Rust rewrite of the [HADES](https://github.com/toddwbucy) knowledge graph system. HADES is a semantic graph platform built on ArangoDB that ingests, embeds, and interconnects research papers, source code, equations, and experimental data. HADES-Burn replaces the original Python implementation with native Rust — retaining Python only for GPU-bound ML inference (embedding and extraction services).

## Architecture

```
┌──────────────────────────────────────────────────────┐
│                  hades (CLI)                     │
│          14 top-level commands, ~85 subcommands       │
└──────────────┬───────────────────────┬───────────────┘
               │                       │
   ┌───────────▼──────────┐  ┌────────▼────────────┐
   │     hades-core        │  │    hades-proto       │
   │  ArangoDB client,     │  │  gRPC/protobuf for   │
   │  graph engine,        │  │  Persephone provider  │
   │  code analysis,       │  │  protocol (embed +    │
   │  pipeline, chunking   │  │  extraction)          │
   └───────────┬───────────┘  └────────────────────────┘
               │
   ┌───────────▼──────────┐
   │   hades-prefetch      │
   │  Async graph-aware    │
   │  batch prefetcher     │
   │  for GNN training     │
   └───────────────────────┘
```

**External services** (Python, GPU-bound):
- **Embedder** — Jina V4 via gRPC on `/run/hades/embedder.sock`
- **Extractor** — Docling VLM via gRPC on `/run/hades/extractor.sock`

**Database**: ArangoDB over Unix socket (`/run/arangodb3/arangodb.sock`)

## Build

```bash
cargo build              # debug
cargo build --release    # release
cargo test               # ~350 tests
cargo clippy             # lint
```

Binary: `target/debug/hades` (or `target/release/hades`)

Requires Rust edition 2024 (nightly or stable 1.85+).

## Usage

```bash
# Target a database (all commands)
hades --db NestedLearning db query "attention mechanism" -n 5

# Semantic search (vector + hybrid)
hades --db mydb db query "graph neural networks" --hybrid -n 10

# Get a document
hades --db mydb db get papers arxiv_2501_00663

# Graph traversal
hades --db mydb db graph traverse "papers/arxiv_2501_00663" -d outbound --max-depth 3

# Ingest a codebase (AST analysis + optional embedding)
hades --db mydb codebase ingest /path/to/project --lang rust

# Initialize a database schema
hades --db mydb db schema init --seed nl

# Materialize typed edges from cross-reference fields
hades --db mydb db graph materialize --dry-run

# Task management
hades --db bident_burn task list
hades --db bident_burn task create --title "New feature" --priority high

# Start the daemon (Unix socket query server)
hades daemon
```

All output is JSON to stdout. Progress and logs go to stderr.

## Command Groups

| Group | Commands | Description |
|-------|----------|-------------|
| `db` | query, aql, get, list, count, insert, update, delete, purge, create, collections, databases, export, health, stats, recent, check, create-database, create-index, index-status, backfill-text | Database CRUD, search, and administration |
| `db graph` | traverse, shortest-path, neighbors, create, list, drop, materialize | Graph traversal and edge materialization |
| `db schema` | init, list, show, version | Runtime ontology management |
| `codebase` | ingest, update, stats, validate | AST-level code ingestion with import resolution |
| `task` | list, show, create, update, close, start, review, approve, block, unblock, handoff, handoff-show, context, log, sessions, dep, usage, graph-integration | Kanban-style task lifecycle |
| `embed` | text, service (start/stop/status), gpu (status/list) | Embedding generation and service management |
| `graph-embed` | train, embed, neighbors, update | GraphSAGE/RGCN structural embeddings |
| `smell` | check, verify, report | Code smell detection and compliance |
| `arxiv` | sync, sync-status | arXiv paper synchronization |
| Top-level | status, orient, extract, ingest, link, daemon | System, ingestion, and daemon |

## Key Features

### Codebase Ingestion
AST-level analysis for Rust (syn + rust-analyzer) and Python (rustpython-parser). Extracts symbols, qualified names, call hierarchy, impl-trait relationships, and cross-file import edges. Token-aware chunking respects symbol boundaries. Optional vector embedding via Jina V4.

### Runtime Schema (Ontology as Data)
Database ontologies are stored in the `hades_schema` collection, not in source code. Seeds (`--seed nl`, `--seed empty`) provision schemas. Edge materialization and graph creation read from the runtime schema, falling back to NL statics for databases without one.

### Daemon Protocol
Unix socket server at `/run/hades/hades.sock` with session-based access control. Three tiers: Agent (safe reads + task management), Internal (system diagnostics), Admin (writes + DDL + raw AQL). See [docs/daemon-protocol.md](docs/daemon-protocol.md).

### Model Operation Vocabulary
Closed set of operations designed for AI model agents. Models never write raw AQL — HADES provides bounded operations that translate internally. Naming follows the training distribution of 24-32B parameter models. See [docs/model-operation-vocabulary.md](docs/model-operation-vocabulary.md).

## Configuration

HADES-Burn reads configuration from `~/.config/hades/config.yaml` with environment variable overrides:

| Variable | Purpose |
|----------|---------|
| `HADES_DATABASE` | Default database name |
| `HADES_ARANGO_HOST` | ArangoDB host (default: Unix socket) |
| `HADES_USE_GPU` | Enable/disable GPU for embeddings |
| `HADES_EMBEDDER_SOCKET` | Embedder service socket path |
| `HADES_EXTRACTOR_SOCKET` | Extractor service socket path |

## Documentation

| Document | Description |
|----------|-------------|
| [Daemon Protocol](docs/daemon-protocol.md) | Wire protocol, sessions, access tiers, all 44 commands |
| [Model Operation Vocabulary](docs/model-operation-vocabulary.md) | Closed operation set for AI model agents |
| [Codebase Graph Ontology](docs/codebase-graph-ontology.md) | Universal code ontology — collections, edges, named graph |
| [OG-RAG Integration](docs/og-rag-report.md) | Ontology-grounded hypergraph retrieval analysis and integration proposal |
| [Agent Memory Design](docs/design-agent-memory-and-system-prompt.md) | Cross-project design for agent memory and system prompt bootstrap |

## Project Status

| Metric | Value |
|--------|-------|
| Rust source files | 101 |
| Lines of Rust | ~36,000 |
| Tests | ~350 |
| PRs merged | 57 |
| Commits | 231 |
| CLI subcommands | ~85 (all native Rust) |

**Still falling through to Python** (2 remaining):
- Cross-encoder reranking in `db query`
- Structural graph fusion in `db query`

Once these are native, the Python CLI dependency drops entirely.

## License

[MIT](LICENSE)
