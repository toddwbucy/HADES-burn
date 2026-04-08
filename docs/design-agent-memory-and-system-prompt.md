# Design: Agent Memory, System Prompt, and Universal Ontologies

**Status**: Draft / Request for Discussion  
**Date**: 2026-04-08  
**Scope**: HADES-Burn + weaver-tools integration  
**Author**: Todd Bucy + Claude (HADES-Burn session)

---

## Problem

Right now, weaver-tools builds system prompts in code (`SystemPromptBuilder` in
weaver-query) and has no persistent agent memory. Each session is a flat JSON
file of conversation history. When the model loads, everything it knows comes
from a hardcoded prompt and whatever fits in the context window.

This creates several problems:

1. **Updating agent behavior requires code changes.** Modifying the system prompt
   means editing Rust source, recompiling, and redeploying.
2. **No persistent memory across sessions.** The agent forgets everything between
   conversations. Session JSON stores raw messages but no structured knowledge.
3. **No per-agent customization.** Every agent gets the same prompt from the same
   code path. Specializing an agent means code branching.
4. **Context window waste.** The full system prompt loads every time, even when
   most of it is irrelevant to the current task.

## Proposal: System Prompt as Data, Memory as Graph

### Core Idea

The system prompt and agent memory should live in HADES, not in source code.
The harness (weaver-tools) becomes a dumb pipe: it knows the agent's identity,
the HADES socket, and how to execute one bootstrap query. Everything else — the
agent's instructions, constraints, knowledge, and history — comes from the graph.

### The Bootloader Analogy

Think of the system prompt like the Linux boot process:

- **initramfs (harness)**: Just enough to mount the filesystem. Agent ID, HADES
  connection info, and the bootstrap query.
- **Root filesystem (memory database)**: The actual system prompt, agent
  preferences, learned constraints, domain knowledge, session history.
- **init process (first turn)**: Agent queries its memory DB, loads relevant
  context, and begins work with full awareness.

The harness hardcodes only:
```
agent_id = "agent-007"
hades_socket = "/run/hades/hades.sock"
hades_db = "agent_007_memory"

system_prompt = hades --db {hades_db} db read system_prompts/active
```

Then passes that string to the model API as the system message. Done.

### What This Enables

- **Update agent behavior with a database write.** No recompilation. No PR.
  `hades --db agent_memory db write system_prompts/active '{...}'`
- **Version and rollback prompts.** Document revisions in ArangoDB give you
  history. A/B testing is swapping which document `active` points to.
- **Per-agent specialization.** Each agent has its own memory DB with its own
  system prompt. No code branching.
- **Rotating models.** Point a new model at the same memory DB. The knowledge
  persists in the graph, not in the weights or the prompt template.

---

## Three Universal Ontologies

HADES already has a two-tier ontology model:

1. **Universal layer** — always present, same in every database
2. **Subject layer** — per-database, domain-specific, stored in `hades_schema`

We've established that **code has a universal ontology** (`codebase_files`,
`codebase_symbols`, `codebase_imports_edges`, etc.) because code is code
regardless of domain. A protein folding codebase and a commodities trading bot
share the same structural primitives.

The same logic applies to **agent memory**. Remembering is remembering regardless
of domain. The primitives of memory don't change between a protein folding agent
and a corn futures agent.

This gives us three universal ontologies, each implemented as a seed:

| Seed | Scope | When Applied |
|------|-------|-------------|
| `--seed codebase` | Any DB with ingested code | `codebase ingest` ensures collections |
| `--seed memory` | Any agent memory DB | Harness provisions DB at agent creation |
| `--seed {domain}` | Domain-specific subject matter | Agent or human defines at runtime |

### Memory Ontology — Primitives to Discuss

These are starting points, not final. The question for weaver-tools: what does
the harness need to read/write, and what does the agent need to query?

**Candidate vertex collections:**
- `system_prompts` — versioned system prompt documents, `active` pointer
- `observations` — facts the agent has learned (with confidence, source, timestamp)
- `decisions` — choices made and their rationale
- `preferences` — user/environment preferences learned over time
- `sessions` — structured session summaries (not raw message dumps)
- `tasks` — persistent task state across sessions

**Candidate edge collections:**
- `memory_source_edges` — observation/decision → source document/conversation
- `memory_supersedes_edges` — newer observation → older observation it replaces
- `memory_relates_edges` — general semantic links between memory nodes

**Candidate named graph:**
- `agent_memory` — composes all memory edges for traversal

**Open questions:**
- Should session history (raw messages) live in the graph or stay as flat files?
- How much of the system prompt is structured (parseable sections) vs opaque text?
- Does the agent write to its own memory, or does the harness mediate all writes?
- Should there be a `memory_schema` separate from `hades_schema`, or is the
  memory ontology just another seed written to the same `hades_schema` collection?

---

## Integration Points with weaver-tools

### Current State

weaver-tools currently integrates with HADES via CLI shell-outs:
- `HadesIngestTool`, `HadesSearchTool`, `HadesSymbolTool`, `HadesTraverseTool`
- Each tool invokes the `hades` binary, parses JSON stdout
- Configuration: database name + binary path in `HadesConfig`
- `SystemPromptBuilder` dynamically adds HADES instructions if tools are present

### What Changes

1. **Bootstrap query**: Before building `ModelRequest`, the harness reads the
   system prompt from the agent's memory DB instead of calling
   `default_system_prompt()`.

2. **Session lifecycle**: At session start, the agent (or harness) queries memory
   for relevant context. At session end, the harness writes a structured summary
   back to memory.

3. **Agent provisioning**: Creating a new agent means:
   ```
   hades db create agent_{id}_memory
   hades --db agent_{id}_memory db schema init --seed memory
   hades --db agent_{id}_memory db write system_prompts/active '{...}'
   ```

4. **Memory tools**: New HADES tools the agent can use to read/write its own
   memory: `HadesMemoryReadTool`, `HadesMemoryWriteTool`. These are distinct
   from the code graph tools — they operate on the agent's memory DB.

### What Doesn't Change

- The tool trait and dispatch system — memory tools are just more `Tool` impls
- The query loop engine — still streams, still batches tool calls
- The provider layer — model API calls are unchanged
- The permission system — memory writes go through the same permission callback

---

## Dependency Order

This doesn't all need to happen at once. Suggested sequence:

1. **HADES: `--seed memory` implementation** — define the memory ontology, add it
   as a seed option alongside `nl` and `empty`. Uses the same `hades_schema`
   infrastructure from Phase 1-2.

2. **HADES: `db read` / `db write` for system prompts** — ensure the CLI can
   read/write system prompt documents cleanly.

3. **weaver-tools: bootstrap from HADES** — modify `SystemPromptBuilder` to
   optionally load the system prompt from a HADES memory DB instead of building
   it in code. Fallback to code-built prompt when no DB is configured.

4. **weaver-tools: memory tools** — add `HadesMemoryReadTool` /
   `HadesMemoryWriteTool` so the agent can query and update its own memory
   during a session.

5. **weaver-tools: session-end writeback** — at session end, write a structured
   summary to the memory graph.

---

## HADES-Burn Functional Parity Status

weaver-tools currently shells out to the `hades` CLI binary. Any integration
work depends on what's actually implemented in the Rust CLI vs what still falls
through to Python. Here's the current state as of 2026-04-08.

### Fully Implemented in Rust (~85 commands)

| Group | Commands | Notes |
|-------|----------|-------|
| **System** | `status`, `orient`, `extract` | |
| **Ingest** | `ingest`, `link` | |
| **arXiv** | `arxiv sync`, `arxiv sync-status` | Legacy — arxiv becomes metadata field post-cutover |
| **DB Read** | `query`, `get`, `count`, `list`, `collections`, `databases`, `check`, `recent`, `health`, `stats`, `export`, `index-status`, `aql` | `query` handles vector+hybrid natively |
| **DB Write** | `insert`, `update`, `delete`, `purge`, `create`, `create-database`, `create-index`, `backfill-text` | |
| **DB Graph** | `traverse`, `shortest-path`, `neighbors`, `list`, `create`, `drop`, `materialize` | `materialize` reads from RuntimeSchema (Phase 2) |
| **DB Schema** | `init`, `list`, `show`, `version` | New — runtime ontology management (Phase 1-2) |
| **DB Search** | `search` | Vector + hybrid search |
| **Task** | `list`, `show`, `create`, `update`, `close`, `start`, `review`, `approve`, `block`, `unblock`, `handoff`, `handoff-show`, `context`, `log`, `sessions`, `dep`, `usage`, `graph-integration` | Full kanban lifecycle |
| **Embed** | `text`, `service status/start/stop`, `gpu status/list` | Jina V4 via gRPC |
| **Codebase** | `ingest`, `update`, `stats`, `validate` | AST-level chunking, import edge resolution |
| **Smell** | `check`, `verify`, `report` | Code smell compliance |
| **Graph-Embed** | `train`, `embed`, `neighbors`, `update` | GraphSAGE/RGCN training pipeline |
| **Daemon** | `daemon` | Unix socket wire protocol with access tiers |

### Still Falling Through to Python

| Feature | Why | Impact on Integration |
|---------|-----|----------------------|
| Cross-encoder reranking in `db query` | Needs Rust cross-encoder client | Low — agents rarely need reranking directly |
| Structural graph fusion in `db query` | Complex query planner | Low — `db aql` covers the same queries |

### Key Architectural Facts for Integration

- **Strangler-fig pattern**: Unmatched CLI invocations pass through to the
  Python `hades` binary. Once the last two features are native, the Python
  dependency drops entirely.
- **Daemon wire protocol**: The Rust daemon (`/run/hades/hades.sock`) accepts
  JSON commands over Unix socket with session-based access tiers (Admin,
  Internal, Agent). Agent sessions restrict which commands are available.
- **RuntimeSchema (just merged)**: `hades_schema` collection stores per-database
  ontology. `RuntimeSchema::load()` reads from DB, falls back to NL statics.
  Seeds: `--seed nl`, `--seed empty`. Adding `--seed memory` is straightforward.
- **All output is JSON to stdout**, logs to stderr. This is the interface
  weaver-tools already consumes via shell-out.

### What This Means for the Memory/Prompt Work

1. **`db read` and `db write` already exist.** The harness can read a system
   prompt document today: `hades --db agent_memory db get system_prompts active`.
   Writing: `hades --db agent_memory db insert system_prompts '{...}'`.

2. **`db schema init --seed memory` is the main new work on the HADES side.**
   The seed infrastructure is built. We just need to define the memory ontology
   and add it as a seed option.

3. **The daemon's agent access tier** already restricts what an agent can do.
   Memory writes would go through the same permission model — the agent writes
   to its own memory DB, the daemon enforces boundaries.

4. **No native Rust client needed yet.** weaver-tools shells out to the CLI
   binary, and that's fast enough for bootstrap reads and session-end writes.
   A native client is a future optimization, not a blocker.

---

## Decision Needed from weaver-tools Session

This document originated in the HADES-Burn session. The memory ontology
primitives, system prompt document structure, and bootstrap flow need input from
the weaver-tools side:

- What does `SystemPromptBuilder` need that can't come from a flat document?
  (e.g., dynamic tool schema injection — does that stay in code?)
- What's the right boundary between harness responsibility and agent autonomy
  for memory writes?
- Is the HADES CLI shell-out sufficient for the bootstrap read, or does this
  motivate a native Rust client?

This is a conversation starter, not a spec. Let's iterate.
