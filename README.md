# HADES-Burn

[![CI](https://github.com/toddwbucy/HADES-burn/actions/workflows/ci.yml/badge.svg)](https://github.com/toddwbucy/HADES-burn/actions/workflows/ci.yml)

## *for ontology-grounded context engineering*

**HADES-Burn** (High-speed ArangoDB Data Embedding System) is research infrastructure for low-latency retrieval-augmented generation over persistent, ontology-grounded knowledge graphs. It was developed as the context-management substrate for the Nested Learning research program — a year-long effort to implement and validate published work on nested-optimization memory systems from the Mirrokni et al. research group (arXiv:2512.24695, arXiv:2501.00663) — and has been empirically validated through daily use as the retrieval backend for both local inference (24–32B Qwen and Mistral variants) and commercial coding agents (Claude Code) during the project's own development.

The design premise under investigation is that context engineering — the structured construction of what an agent retrieves and reads on each turn — is more often the binding constraint on sustained agent coherence than raw model capability. HADES-Burn pursues this through an ontology-grounded schema and a closed operation vocabulary that together produce traceable, structured retrieval: the kind of context discipline that on-premises agent workflows require to remain coherent across long-horizon tasks.

The context-engineering methodology and schema this rests on — how a foundation document and a codebase become an axiom-gated ("IS / IS-NOT") knowledge graph in which concepts, smells, and code must *earn* their place by tracing to ratified axioms, and in which non-connection becomes a queryable signal — is formalized in **[Bastion/graph-methodology.md](Bastion/graph-methodology.md)**.

HADES is one *species* of a more general method — a **bastion**: a project's canon stood up as a context graph and wired into the code and release pipeline as a governance layer, on any graph-capable backend. The backend-agnostic founding document is **[Bastion/foundation/the-bastion.md](Bastion/foundation/the-bastion.md)** (the architecture), serving the philosophical foundation in **[Bastion/foundation/bastion-of-context.md](Bastion/foundation/bastion-of-context.md)** (the *why*). `graph-methodology.md` is the ArangoDB reference implementation of that method.

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

**Tier-aware command dispatch.** The query daemon at `/run/hades/hades.sock` classifies commands into three tiers — Agent (safe reads and task management), Internal (diagnostics), and Admin (writes, DDL, raw AQL) — and supports opt-in client self-restriction via the request's `session` field. An AI-agent harness can declare `"session": "agent"` to have the daemon reject any commands above the Agent tier, preventing accidental admin actions from a model that was only meant to read. The tier metadata is documentation as much as enforcement: it makes the agent-safe command set inspectable. *Security against malicious clients is not the goal here; that is enforced at the ArangoDB layer via ACL grants on the dedicated `hades` ArangoDB user (configured at install time).* See [docs/daemon-protocol.md](docs/daemon-protocol.md).

**Tiered multilingual code ingestion.** Rust uses `syn` plus rust-analyzer, Go uses Tree-sitter plus gopls, Python uses `rustpython-parser`, and C, C++, and CUDA use libclang. C-family ingestion understands namespaces, templates, overloads, calls, and CUDA kernel launches; it can consume a project's `compile_commands.json`. Registered Tree-sitter grammars provide lower-fidelity structural coverage when a dedicated analyzer is unavailable or fails. Every artifact records `semantic`, `structural`, or `text` provenance, and re-ingest will not replace a higher tier unless `--allow-analysis-downgrade` is explicit.

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

### Pre-release verification

Before installing a new binary, run the end-to-end pipeline smoke test
against it. It exercises full pipelines (tasks, document ingest, codebase
ingest, drift→retire→prune, semantic search) on a dedicated
`bident_burn_smoke` database and asserts the graph invariants that past
regressions actually violated — foreign keys on chunks/embeddings,
input canonicalization, the honest batch envelope, `--force` stability:

```
HADES_BIN=target/release/hades ./scripts/bident_burn_smoke.sh
```

Requires live ArangoDB and the embedder service; never touches
`bident_burn` or any production database. `scripts/cli_audit.sh` remains
the command-level companion ("does every command run"); the smoke test
answers "is the graph right".

## Install

The install procedure below has been validated end-to-end on a fresh
Ubuntu 24.04 environment via the container harness at
`scripts/install/test/` (run `bash scripts/install/test/run-install.sh`
inside the harness image; see the script header for details). The
systemd-unit portion is verified only via the harness's manual-daemon
path; the real-VPS systemd flow is documented but tested separately.

### 0. Prerequisites

A clean Linux machine (tested on Ubuntu 24.04; should work on any
recent Debian-family distro). You will need root.

**Install ArangoDB** following the [official docs](https://docs.arangodb.com/3.12/operations/installation/linux/).
The short version for Debian/Ubuntu:

```bash
echo 'deb https://download.arangodb.com/arangodb312/DEBIAN/ /' \
  | sudo tee /etc/apt/sources.list.d/arangodb.list
curl -fsSL https://download.arangodb.com/arangodb312/DEBIAN/Release.key \
  | sudo gpg --dearmor -o /usr/share/keyrings/arangodb.gpg
sudo apt-get update && sudo apt-get install -y arangodb3
```

> **Note (2026-05):** ArangoDB's signing key on their `arangodb312` deb
> repo is currently expired (`EXPKEYSIG`). If `apt-get update` fails
> with that error, either fetch the latest key from the ArangoDB
> downloads page, or temporarily set `deb [trusted=yes] ...` in the
> sources line as a workaround. Their key rotation is out of HADES's
> control.

During `apt-get install` you'll be prompted to set a root password
for ArangoDB — note it, you'll use it in step 2.

**Install Rust** (edition 2024, stable 1.85+):

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
. ~/.cargo/env
```

**Install `protoc`** (needed by `hades-proto`'s build script):

```bash
sudo apt-get install -y protobuf-compiler
```

### 1. Build the binary

From a checkout of this repo:

```bash
cargo build --release
```

This produces `target/release/hades`.

### 2. Create the ArangoDB user

HADES connects to ArangoDB as a dedicated `hades` user (not `root`).
Write restrictions on specific databases are enforced by ArangoDB
ACLs on this user — HADES has no source-level allowlist. Bootstrap
with arangosh, using the root password you set during install:

```bash
arangosh --server.endpoint unix:///run/arangodb3/arangodb.sock \
         --server.username root
# (enter root password at the prompt, then in the arangosh REPL:)
> const users = require("@arangodb/users");
> users.save("hades", "<pick-a-password>");
> users.grantDatabase("hades", "_system", "rw");
> users.grantDatabase("hades", "*", "rw");   // default for new DBs
> // Tighten specific production databases as needed:
> // users.grantDatabase("hades", "<production-db-name>", "ro");
```

Note the password you set for `hades` — you'll use it in step 5.

### 3. Install the systemd users + tmpfiles

These create the Unix `hades` user and group, plus the `/run/hades`
runtime directory. They must run **before** step 4 because the config
file's group ownership references `hades`.

```bash
sudo install -m 644 services/systemd/hades-sysusers.conf  /etc/sysusers.d/hades.conf
sudo install -m 644 services/systemd/hades-tmpfiles.conf  /etc/tmpfiles.d/hades.conf
sudo systemd-sysusers
sudo systemd-tmpfiles --create
```

After this, `getent group hades` should show the group exists.

### 4. Install the binary

```bash
sudo install -m 755 target/release/hades /usr/local/bin/hades
mkdir -p ~/.local/bin && install -m 755 target/release/hades ~/.local/bin/hades
```

The systemd daemon execs `/usr/local/bin/hades`; interactive shells
usually pick up `~/.local/bin/hades` first. If you only update one,
the other will silently run an older build.

### 5. Install the system config

```bash
sudo mkdir -p /etc/hades
sudo install -m 640 -o root -g hades config/hades.yaml /etc/hades/hades.yaml
```

The daemon searches `/etc/hades/hades.yaml` after the in-repo paths;
the username (`hades`) and socket paths live there. Passwords are
never stored in YAML — they come from the daemon's env (step 6). The
username can also be overridden out-of-band with the `ARANGO_USERNAME`
env var, parallel to `ARANGO_PASSWORD` / `ARANGO_HOST` / `ARANGO_PORT` /
`ARANGO_RO_SOCKET` / `ARANGO_RW_SOCKET`.

### 6. Set the daemon environment

`/etc/hades/daemon.conf` is sourced by `hades-daemon.service` via
`EnvironmentFile=`. Create it with:

```bash
sudo tee /etc/hades/daemon.conf <<'CONF'
ARANGO_PASSWORD=<password-you-set-for-the-hades-user>
HADES_DATABASE=_system
ARANGO_RO_SOCKET=/run/arangodb3/arangodb.sock
ARANGO_RW_SOCKET=/run/arangodb3/arangodb.sock
HADES_EMBEDDER_SOCKET=http://localhost:8087/v1
CONF
sudo chown root:hades /etc/hades/daemon.conf
sudo chmod 640 /etc/hades/daemon.conf
```

The daemon needs `HADES_DATABASE` set because `ArangoPool::from_config`
opens a connection at startup; `_system` is the right default (always
present, neutral) and dispatched commands override the target
per-request.

### 7. Enable and start the daemon

```bash
sudo install -m 644 services/systemd/hades-daemon.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now hades-daemon.service
sudo systemctl status hades-daemon.service --no-pager
```

You should see `Active: active (running)` and a log line like
`daemon listening socket="/run/hades/hades.sock"`.

### 8. Verify

`hades db stats` opens its own ArangoDB connection (it doesn't proxy
through the daemon socket), so the CLI needs the `hades` user's
password in its environment:

```bash
export ARANGO_PASSWORD='<password-you-set-for-the-hades-user>'
hades --db _system db stats
```

A clean JSON response listing the `_system` database means
authentication works and your CLI can reach ArangoDB. The
`ARANGO_PASSWORD` export takes precedence over anything in
`hades.yaml`; keep it in your shell environment (or in a profile
file restricted to your user) for ongoing CLI use.

The `hades-embedder.service` and `hades-extractor.service` units cover
the optional embedder (Jina V4) and Docling extractor services —
install only if you're using `hades embed`, `hades codebase ingest --embed`,
or `hades extract`. Both require additional Python dependencies and (for
the embedder) a GPU; see `services/README.md` for details.

### Updating the daemon's ArangoDB password

1. Rotate the password in arangosh:

```javascript
require("@arangodb/users").replace("hades", "<new-password>");
```

2. Edit `/etc/hades/daemon.conf` and replace the `ARANGO_PASSWORD=` value
(file is `root:hades 640`, so use `sudoedit` or `sudo $EDITOR`).
3. Restart the daemon:

```bash
sudo systemctl restart hades-daemon.service
sudo systemctl status hades-daemon.service --no-pager
```

For interactive CLI use (outside the daemon), export `ARANGO_PASSWORD` in your
shell — it takes precedence over anything in YAML.

## Usage

```
hades --db <name> db query "attention mechanism" -n 5
hades --db <name> db graph traverse "papers/example_paper_key" -d outbound --max-depth 3
hades --db <name> codebase ingest /path/to/project --lang rust
hades --db <name> codebase ingest /path/to/cuda-project --compile-commands /path/to/build
hades --db <name> db schema init --seed nl
hades daemon
```

For C/C++/CUDA, `--compile-commands` accepts either the JSON file or its containing directory. If omitted, HADES searches source ancestors and their `build/` directories. One database may contain files from every supported language; language detection and analysis happen per file.

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
| [Bastion/foundation/bastion-of-context.md](Bastion/foundation/bastion-of-context.md) | Foundation Layer 1 — the philosophical *why*: cathedral / bazaar / bastion, the twelve load-bearing principles |
| [Bastion/foundation/the-bastion.md](Bastion/foundation/the-bastion.md) | Foundation Layer 2 — the backend-agnostic architecture: invariants, the two operating modes, the construction layer (wall / scaffolding / delta ledger), code-against-documentation conformance, the backend contract, the de-ratification rite |
| [Bastion/foundation/bastion-playbook.md](Bastion/foundation/bastion-playbook.md) | The operational playbook — how to stand up or retrofit a bastion in a Claude Code session, phase by phase, greenfield and brownfield |
| [Bastion/graph-methodology.md](Bastion/graph-methodology.md) | Foundation Layer 3 — the ArangoDB/HADES reference implementation of the bastion method |
| [RESEARCH_GOALS.md](RESEARCH_GOALS.md) | Research questions, context-management framing, measurement points |
| [CHANGELOG.md](CHANGELOG.md) | Release history and convention for adding entries |
| [docs/specs/workstation-specific-tests.md](docs/specs/workstation-specific-tests.md) | Convention for integration tests that depend on workstation-specific resources |
| [docs/daemon-protocol.md](docs/daemon-protocol.md) | Wire protocol, session model, access tiers, command reference |
| [docs/model-operation-vocabulary.md](docs/model-operation-vocabulary.md) | Closed operation set for AI model agents |
| [docs/codebase-graph-ontology.md](docs/codebase-graph-ontology.md) | Universal code ontology — collections, edges, named graph |
| [docs/og-rag-report.md](docs/og-rag-report.md) | Ontology-grounded hypergraph retrieval analysis and integration proposal |
| [docs/design-agent-memory-and-system-prompt.md](docs/design-agent-memory-and-system-prompt.md) | Cross-project design for agent memory and system prompt bootstrap |

## License

Licensed under the [Apache License, Version 2.0](LICENSE). See `LICENSE` for the full text.
