# Model-Facing Operation Vocabulary

**Status:** v1.0
**Date:** 2026-04-07
**Scope:** Closed operation set for AI model agents interacting with HADES over the daemon socket
**Relates to:** task_6e3f1c (Model-facing operation vocabulary audit), task_149770 (Codebase schema ontology)

---

## 1. Purpose

AI model agents (24-32B Mistral, Qwen, etc.) interact with the HADES knowledge graph through the daemon socket at `/run/hades/hades.sock`. This document defines the **closed vocabulary** of operations they may invoke.

### Core Principle

> Models never write raw AQL. HADES provides bounded operations that translate internally to database queries. Limited actions = limited damage surface.

The daemon enforces this boundary via `AccessTier` classification on every `DaemonCommand` variant. Commands not in the `Agent` tier are rejected for model sessions.

### Naming Alignment

All operation names use vocabulary well-represented in the training distribution of 24-32B parameter models:

- **Standard CRUD**: `get`, `list`, `count`, `check`, `create`, `update`, `close`
- **Standard graph**: `traverse`, `neighbors`, `shortest_path`
- **Standard search**: `query` (semantic/hybrid), `embed` (vector generation)
- **Domain-specific but common**: `orient`, `handoff`, `verify`, `report`

---

## 2. Access Tiers

Every daemon command is classified into exactly one tier:

| Tier | Who | What | Count |
|------|-----|------|-------|
| **Agent** | Model/AI agents | Bounded reads, task management, semantic search, graph traversal, code quality | 36 |
| **Admin** | Human operators | Raw AQL, schema mutation, unbounded writes, destructive operations | 9 |
| **Internal** | Any caller | System diagnostics, health checks — not part of model vocabulary | 5 |

**Enforcement**: `DaemonCommand::access_tier()` returns the tier. The daemon rejects `Admin` commands from agent sessions with `WriteDenied`.

---

## 3. Agent Operations (Model-Safe)

### 3.1 Context & Orientation

| Command | Parameters | Returns | Purpose |
|---------|-----------|---------|---------|
| `orient` | `collection?` | Database metadata, collection schemas, sample documents | First call — understand what data exists |
| `codebase.stats` | — | Collection counts for all 8 codebase collections | Codebase graph health at a glance |

### 3.2 Semantic Search

| Command | Parameters | Returns | Purpose |
|---------|-----------|---------|---------|
| `db.query` | `text`, `limit?`, `collection?`, `hybrid?` | Ranked document list with scores | Primary search — vector similarity with optional hybrid (BM25 + vector) |
| `embed.text` | `text` | Embedding vector, model name, dimension | Generate embedding for downstream use |

### 3.3 Document Retrieval

| Command | Parameters | Returns | Purpose |
|---------|-----------|---------|---------|
| `db.get` | `collection`, `key` | Single document | Fetch by known key |
| `db.list` | `collection?`, `limit?`, `paper?` | Document array | Browse with pagination |
| `db.count` | `collection` | Integer count | Collection size |
| `db.check` | `document_id` | Existence flag + metadata | Verify document exists before acting |
| `db.recent` | `limit?` | Recently added documents | Discover new content |

### 3.4 Graph Traversal

| Command | Parameters | Returns | Purpose |
|---------|-----------|---------|---------|
| `db.graph.traverse` | `start`, `direction?`, `min_depth?`, `max_depth?`, `limit?`, `graph?` | Vertices and edges along paths | Walk the graph from a starting point |
| `db.graph.neighbors` | `vertex`, `direction?`, `limit?`, `graph?` | Adjacent vertices | One-hop exploration |
| `db.graph.shortest_path` | `source`, `target`, `direction?`, `graph?` | Path vertices and edges | Find connection between two nodes |

**Defaults**: `direction=outbound` for traverse, `direction=any` for neighbors/shortest_path, `min_depth=1`, `max_depth=1`.

### 3.5 Structural Embeddings

| Command | Parameters | Returns | Purpose |
|---------|-----------|---------|---------|
| `graph_embed.embed` | `node_id` | Embedding vector | Fetch pre-computed GraphSAGE embedding |
| `graph_embed.neighbors` | `node_id`, `limit?` | Structurally similar nodes | Find graph-structure neighbors (not semantic) |

### 3.6 Task Management

Models manage their own work through the full task lifecycle:

| Command | Parameters | Returns | Purpose |
|---------|-----------|---------|---------|
| `task.list` | `status?`, `type?`, `parent?`, `limit?` | Task array | Browse tasks with filtering |
| `task.show` | `key` | Single task | Read task details |
| `task.create` | `title`, `description?`, `type?`, `parent?`, `priority?`, `tags[]` | Created task | Create new work item |
| `task.update` | `key`, `title?`, `description?`, `priority?`, `status?`, `add_tags[]`, `remove_tags[]` | Updated task | Modify task fields |
| `task.close` | `key`, `message?` | Closed task | Mark work complete |
| `task.start` | `key` | Started task | Begin work |
| `task.review` | `key`, `message?` | Task in review | Request human review |
| `task.approve` | `key`, `human?` | Approved task | Approve completion |
| `task.block` | `key`, `message?`, `blocker?` | Blocked task | Flag impediment |
| `task.unblock` | `key` | Unblocked task | Clear impediment |
| `task.handoff` | `key`, `message?` | Handed-off task | Transfer to another agent |
| `task.handoff_show` | `key` | Handoff metadata | Read handoff context |
| `task.context` | `key` | Task + related info | Full context for a task |
| `task.log` | `key`, `limit?` | Execution log entries | Audit trail |
| `task.sessions` | `key` | Session history | Who worked on this and when |
| `task.dep` | `key`, `add?`, `remove?`, `graph?` | Dependency info | Manage task dependencies |
| `task.usage` | — | Resource usage stats | System-wide task metrics |
| `task.graph_integration` | — | Schema info | Task graph integration metadata |

### 3.7 Code Quality

| Command | Parameters | Returns | Purpose |
|---------|-----------|---------|---------|
| `smell.check` | `path`, `verbose?` | Smell violations found | Check file against smell rules |
| `smell.verify` | `path`, `claims[]` | Verification results | Verify compliance claims (CS-NN format) |
| `smell.report` | `path` | Full compliance report | Generate compliance report with embeddings |
| `link_code_smell` | `source_id`, `smell_id`, `enforcement`, `methods[]`, `summary?` | Created edge | Link document to smell compliance node |

---

## 4. Admin Operations (Human-Only)

These operations are **never** available to model agents:

| Command | Why Admin | Risk |
|---------|----------|------|
| `db.aql` | Arbitrary AQL query injection | Can read/write/delete any data, bypass all guardrails |
| `db.insert` | Unbounded document writes | Can create arbitrary documents in any collection |
| `db.update` | Unbounded field mutations | Can corrupt any document |
| `db.delete` | Document deletion | Data loss |
| `db.purge` | Cascading deletion across collections | Broad data loss |
| `db.create_collection` | Schema mutation | Can create arbitrary collections |
| `db.create_index` | Index creation | Performance impact, schema change |
| `db.graph.create` | Graph lifecycle | Structural schema change |
| `db.graph.drop` | Graph deletion | Destructive, potentially drops collections |

### Why `db.aql` Is the Critical Exclusion

Raw AQL is the SQL-injection equivalent for graph databases. A model with `db.aql` access can:
- Read any collection regardless of access controls
- Execute `REMOVE` / `UPDATE` / `INSERT` on any data
- Construct queries that scan the entire database
- Bypass the bounded vocabulary entirely

All agent-tier operations translate to AQL **server-side**, with parameterized queries and bounded result sets. The model never sees or constructs AQL.

---

## 5. Internal Operations (System Diagnostics)

Available to any caller but not part of the model vocabulary — models don't need these:

| Command | Purpose |
|---------|---------|
| `status` | System status, workspace discovery |
| `db.health` | Database connectivity check |
| `db.stats` | Database-wide statistics |
| `db.collections` | List all collections with counts |
| `db.graph.list` | List all named graphs |

---

## 6. Wire Protocol

All commands use the same JSON envelope over the Unix socket:

```json
{
  "command": "db.query",
  "params": {
    "text": "attention mechanism",
    "limit": 10,
    "hybrid": true
  },
  "request_id": "optional-echo-id"
}
```

Response:

```json
{
  "request_id": "optional-echo-id",
  "success": true,
  "data": { ... },
  "error": null,
  "error_code": null
}
```

All parameters use `deny_unknown_fields` — typos and invented parameters are rejected at deserialization.

---

## 7. Naming Audit

### Flagged for Review

| Current Name | Issue | Recommendation | Priority |
|-------------|-------|----------------|----------|
| `orient` | Uncommon verb; models may not auto-complete it | Consider alias `describe` or `context` | Low — functional, documented |
| `task.dep` | Abbreviated; models may try `task.dependency` | Consider alias `task.dependency` | Low — works with documentation |
| `link_code_smell` | Inconsistent separator (underscore vs dot) | Consider `smell.link` for consistency | Medium — breaks dot-namespace pattern |
| `graph_embed.embed` | `graph_embed` uses underscore, not dot | Consider `graph.embed.embed` or leave as-is | Low — established pattern |

### Confirmed Good

All other names use standard vocabulary that models handle well:
- CRUD family: `get`, `list`, `count`, `check`, `insert`, `update`, `delete`, `create`
- Graph family: `traverse`, `neighbors`, `shortest_path`
- Search: `query` (overloaded but universal), `embed`
- Task lifecycle: `show`, `start`, `close`, `review`, `approve`, `block`, `unblock`, `handoff`
- Compound names use dots for namespacing: `db.query`, `task.list`, `smell.check`

### Naming Conventions

1. **Dot-separated namespaces**: `domain.action` (e.g., `db.get`, `task.list`)
2. **Verb-based actions**: operations are verbs or verb phrases
3. **No abbreviations** except established ones (`db`, `aql`, `embed`)
4. **Defaults favor safety**: `direction=outbound` (not `any`), limits always bounded

---

## 8. Implementation Reference

- **Access tier enum**: `hades_core::dispatch::AccessTier` (`Agent`, `Admin`, `Internal`)
- **Classification method**: `DaemonCommand::access_tier()` — exhaustive match on all 50 variants
- **Convenience method**: `DaemonCommand::is_agent_safe()` — returns `true` for `Agent` tier
- **Wire protocol**: `DaemonCommand` uses `#[serde(tag = "command", content = "params")]`
- **Param validation**: All param structs use `#[serde(deny_unknown_fields)]`
- **Limit enforcement**: `MAX_LIMIT = 1000` enforced on all paginated operations
