# HADES-Burn

## *for ontology-grounded context engineering*

**HADES-Burn** (High-speed ArangoDB Data Embedding System) is research infrastructure for low-latency retrieval-augmented generation over persistent, ontology-grounded knowledge graphs. It was developed as the context-management substrate for the Nested Learning research program — a year-long effort to implement and validate published work on nested-optimization memory systems from the Mirrokni et al. research group (arXiv:2512.24695, arXiv:2501.00663) — and has been empirically validated through daily use as the retrieval backend for both local inference (24–32B Qwen and Mistral variants) and commercial coding agents (Claude Code) during the project's own development.

The design premise under investigation is that context engineering — the structured construction of what an agent retrieves and reads on each turn — is more often the binding constraint on sustained agent coherence than raw model capability. HADES-Burn pursues this through an ontology-grounded schema and a closed operation vocabulary that together produce traceable, structured retrieval: the kind of context discipline that on-premises agent workflows require to remain coherent across long-horizon tasks.

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

- Embedder — Jina V4 (Qwen2.5-VL-3B + LoRA) via OpenAI-compatible HTTP on `http://localhost:8087/v1` (HADES-owned FastAPI service on dedicated GPU). The contract is the [Persephone Embedding API v1](docs/persephone-embedding-api.md).
- Extractor — Docling VLM via gRPC on `/run/hades/extractor.sock`

**Database**: ArangoDB over Unix socket at `/run/arangodb3/arangodb.sock`.

## Design Decisions Relevant to the Research Questions

**Unix sockets in the hot query path.** The query hot path — CLI client to daemon at `/run/hades/hades.sock`, daemon to ArangoDB at `/run/arangodb3/arangodb.sock` — uses Unix domain sockets. This is the load-bearing latency decision for query workloads and sets the floor for what the rest of the system can achieve. The embedder takes a different transport choice (HTTP-over-TCP localhost) because retrieval ingest is not latency-bound the same way and the OpenAI-compatible HTTP contract is what makes HADES engine-agnostic — see [PE-API v1](docs/persephone-embedding-api.md). The extractor remains on a Unix socket for now.

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

- Jina V4 embedder service exposing the [PE-API v1 HTTP contract](docs/persephone-embedding-api.md) (default `http://localhost:8087/v1`) — required for `hades embed`, `hades codebase ingest --embed`, and hybrid-search queries. Reference implementation in `services/embedding/http_server.py`.

  *Two configs to keep aligned:* the embedder service's **server-side** listen address, GPU device, batch size, and idle timeout come from `/etc/hades/embedder.conf` (sourced by `hades-embedder.service` via `EnvironmentFile=`). HADES's **client-side** endpoint — where the daemon and CLI look for the embedder — comes from `embedding.service.socket` in `hades.yaml` (or the `HADES_EMBEDDER_SOCKET` env var, e.g. in `/etc/hades/daemon.conf`). The two must agree on host:port; changing one without the other silently sends client traffic to the wrong service. The `socket` field accepts `http://`, `https://`, `unix://`, and bare-path endpoint forms (the latter intended for the future `hades-weaver-bridge` Unix-socket adapter).

- Docling extractor service on a Unix socket (default `/run/hades/extractor.sock`) — required for `hades extract` (paper / PDF ingestion).

## Build

```
cargo build              # debug
cargo build --release    # release
cargo test               # ~350 tests
cargo clippy             # lint
```

Binary: `target/debug/hades` or `target/release/hades`.

## Install

> **WIP.** A proper installer is not yet wired up. The steps below document the
> manual bootstrap that the systemd unit (`services/systemd/hades-daemon.service`)
> and the CLI both depend on. Treat this as a checklist, not a script.
> See `scripts/install/setup-arangodb-user.sh` for the (untested) ArangoDB-user
> bootstrap sketch.

### 1. Install the binary

```
sudo install -m 755 target/release/hades /usr/local/bin/hades
install -m 755 target/release/hades ~/.local/bin/hades   # for your shell PATH
```

The systemd daemon execs `/usr/local/bin/hades`; interactive shells usually pick
up `~/.local/bin/hades` first. If you only update one, the other will silently
run an older build.

### 2. Create the ArangoDB user

HADES connects to ArangoDB as a dedicated `hades` user (not `root`). Write
restrictions on specific databases are enforced by ArangoDB ACLs on this user —
HADES has no source-level allowlist. Bootstrap with arangosh:

```
const users = require("@arangodb/users");
users.save("hades", "<pick-a-password>");
users.grantDatabase("hades", "_system", "rw");
users.grantDatabase("hades", "*", "rw");   // default for new DBs
// Tighten specific production databases as needed:
// users.grantDatabase("hades", "NestedLearning", "ro");
```

### 3. Install the system config

```
sudo install -m 640 -o root -g hades config/hades.yaml /etc/hades/hades.yaml
```

The daemon searches `/etc/hades/hades.yaml` after the in-repo paths; the username
(`hades`) and socket paths live there. Passwords are never stored in YAML.

### 4. Set the daemon environment

`/etc/hades/daemon.conf` is sourced by `hades-daemon.service` via `EnvironmentFile=`. Minimum required keys:

```
ARANGO_PASSWORD=<password-you-set-for-the-hades-user>
HADES_DATABASE=_system          # bootstrap DB; per-command --db overrides this
ARANGO_RO_SOCKET=/run/arangodb3/arangodb.sock
ARANGO_RW_SOCKET=/run/arangodb3/arangodb.sock
HADES_EMBEDDER_SOCKET=http://localhost:8087/v1
```

The daemon needs `HADES_DATABASE` set because `ArangoPool::from_config` opens a
connection at startup; `_system` is the right default (always present, neutral)
and dispatched commands override the target per-request.

### Updating the daemon's ArangoDB password

1. Rotate the password in arangosh:

```
require("@arangodb/users").replace("hades", "<new-password>");
```

2. Edit `/etc/hades/daemon.conf` and replace the `ARANGO_PASSWORD=` value
(file is `root:hades 640`, so use `sudoedit` or `sudo $EDITOR`).
3. Restart the daemon:

```
sudo systemctl restart hades-daemon.service
sudo systemctl status hades-daemon.service --no-pager
```

For interactive CLI use (outside the daemon), export `ARANGO_PASSWORD` in your
shell — it takes precedence over anything in YAML.

### Systemd units

```
sudo cp services/systemd/hades-daemon.service /etc/systemd/system/
sudo cp services/systemd/hades-sysusers.conf  /etc/sysusers.d/hades.conf
sudo cp services/systemd/hades-tmpfiles.conf  /etc/tmpfiles.d/hades.conf
sudo systemd-sysusers
sudo systemd-tmpfiles --create
sudo systemctl daemon-reload
sudo systemctl enable --now hades-daemon.service
```

The `hades-embedder.service` and `hades-extractor.service` units cover the
optional embedder and Docling services — install only if you're using those
features.

## Usage

```
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
| --- | --- |
| Rust source files | 100 |
| Lines of Rust | ~36,000 |
| Tests | ~350 |
| PRs merged | 66 |
| CLI subcommands | ~85 (all native Rust) |

Not yet ported: `--rerank` (cross-encoder model, deferred until ONNX runtime integration).

## Authorship and Development Process

Authorship of the design and ownership of the resulting codebase reside with the author. Research direction, architectural decisions (Unix-socket transport layer, closed model-operation vocabulary, ontology-grounded schema, three-tier access model, data-sacrosanct operational boundaries), integration with the broader Nested Learning research program, and the dogfooding methodology are the author's work. Implementation was directed by the author with assistance from Claude Code and other contemporary AI coding agents, in the same manner that contemporary software development draws on IDE autocomplete, refactoring tools, and library code.

The project additionally serves as a long-running empirical study of AI-assisted development workflows: the commit and refactor history is itself data on what retrieval and context primitives make AI coding agents effective at sustained engineering work.

## Contributing

The project is open to collaboration from academic and industry researchers working on retrieval latency, local LLM deployment, agent memory systems, or ontology-grounded retrieval. Issues and pull requests are welcome.

## Documentation

| Document | Description |
| --- | --- |
| [RESEARCH_GOALS.md](RESEARCH_GOALS.md) | Research questions, context-management framing, measurement points |
| [docs/daemon-protocol.md](docs/daemon-protocol.md) | Wire protocol, session model, access tiers, command reference |
| [docs/model-operation-vocabulary.md](docs/model-operation-vocabulary.md) | Closed operation set for AI model agents |
| [docs/codebase-graph-ontology.md](docs/codebase-graph-ontology.md) | Universal code ontology — collections, edges, named graph |
| [docs/og-rag-report.md](docs/og-rag-report.md) | Ontology-grounded hypergraph retrieval analysis and integration proposal |
| [docs/design-agent-memory-and-system-prompt.md](docs/design-agent-memory-and-system-prompt.md) | Cross-project design for agent memory and system prompt bootstrap |

## License

Licensed under the [Apache License, Version 2.0](LICENSE). See `LICENSE` for the full text.
