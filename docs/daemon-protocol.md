# HADES Daemon Protocol Specification

**Version:** 2.0  
**Updated:** 2026-04-08  
**Socket:** `/run/hades/hades.sock` (Unix domain socket, configurable)

## Overview

The HADES daemon exposes the CLI's query, CRUD, and management surface over a
persistent Unix domain socket. Local agents (weaver-tools, Persephone sessions)
connect once and issue commands without fork-per-call overhead.

**44 commands** across 8 logical groups, gated by a three-tier access model.

## Wire Format

Length-prefixed JSON framing. Every message (request and response) is:

```text
[4 bytes: big-endian u32 payload length][JSON payload]
```

- **Maximum payload:** 16 MiB (16,777,216 bytes). Messages exceeding this
  limit are rejected with error code `PAYLOAD_TOO_LARGE` without parsing.
- **Encoding:** UTF-8 JSON. No trailing newline inside the payload.
- **Connection model:** one request, one response, serial per connection.
  Pipelining is reserved for a future version (see `request_id`).
- **Strict validation:** All param structs enforce `deny_unknown_fields`.

### Example (hex)

```text
00 00 00 20 {"command":"orient","params":{}}
```

The 4-byte header `0x00000020` = 32, which is the byte length of the
JSON payload that follows.

## Request Schema

```jsonc
{
  // Optional, echoed in response. Reserved for future pipelining.
  "request_id": "abc-123",

  // Optional. Session type: "admin" (default) or "agent".
  // Controls which commands are accessible.
  "session": "agent",

  // Required. One of the command names defined below.
  "command": "db.query",

  // Required. Command-specific parameters (may be empty object).
  "params": {
    "text": "attention mechanism",
    "limit": 10
  }
}
```

| Field        | Type             | Required | Description                               |
|--------------|------------------|----------|-------------------------------------------|
| `request_id` | `string \| null` | no       | Echoed in response; for client correlation |
| `session`    | `string \| null` | no       | `"admin"` (default) or `"agent"`           |
| `command`    | `string`         | yes      | Command name (dotted namespace)            |
| `params`     | `object`         | yes      | Command-specific parameters                |

## Response Schema

```jsonc
{
  "request_id": "abc-123",
  "success": true,
  "data": { ... },
  "error": null,
  "error_code": null
}
```

| Field        | Type             | Present | Description                           |
|--------------|------------------|---------|---------------------------------------|
| `request_id` | `string \| null` | always  | Echoed from request                   |
| `success`    | `bool`           | always  | Whether the command succeeded          |
| `data`       | `object \| null` | always  | Result payload (null on error)         |
| `error`      | `string \| null` | always  | Error message (null on success)        |
| `error_code` | `string \| null` | always  | Machine-readable code (null on success)|

## Sessions and Access Tiers

The daemon implements a three-tier access model. Each command is classified
into exactly one tier. Session type determines the ceiling.

| Session    | Can Execute         | Use Case                    |
|------------|---------------------|-----------------------------|
| `"admin"`  | Agent + Internal + Admin | Human operator, CLI, harness |
| `"agent"`  | Agent only          | AI model agents              |

If omitted, session defaults to `"admin"`.

### Access Tiers

| Tier       | Description                                              |
|------------|----------------------------------------------------------|
| **Agent**  | Safe, bounded reads and task management. No DDL, no raw AQL. |
| **Internal** | System diagnostics and schema introspection.            |
| **Admin**  | Unbounded writes, DDL, raw AQL, schema mutation.         |

An `"agent"` session attempting an Internal or Admin command receives
error code `ACCESS_DENIED`.

## Error Codes

| Code                | Meaning                                             |
|---------------------|-----------------------------------------------------|
| `UNKNOWN_COMMAND`   | `command` field not recognized                       |
| `INVALID_PARAMS`    | Missing, malformed, or out-of-range params           |
| `INVALID_SESSION`   | `session` field is not `"admin"` or `"agent"`        |
| `ACCESS_DENIED`     | Agent session attempted Admin/Internal command        |
| `MALFORMED_JSON`    | Payload is not valid JSON                            |
| `PAYLOAD_TOO_LARGE` | Payload exceeds 16 MiB limit                         |
| `NOT_FOUND`         | Requested document/collection/node does not exist    |
| `QUERY_FAILED`      | ArangoDB query execution error or missing embedding  |
| `SERVICE_ERROR`     | External service error (embedder, etc.)              |
| `READ_ONLY`         | Write attempted on read-only database                |
| `INTERNAL`          | Unexpected server-side error or request timeout      |

---

## Commands — System

### `orient` (Agent)

Metadata-first context orientation for a database.

| Param        | Type     | Default | Description                |
|--------------|----------|---------|----------------------------|
| `collection` | `string` | null    | Focus on specific collection |

### `status` (Internal)

System status — ArangoDB, embedder, sync, config.

| Param     | Type   | Default | Description     |
|-----------|--------|---------|-----------------|
| `verbose` | `bool` | false   | Extended output |

---

## Commands — Database Read

### `db.query` (Agent)

Semantic search over documents (vector similarity + optional hybrid).

| Param        | Type     | Default | Description                        |
|--------------|----------|---------|------------------------------------|
| `text`       | `string` | required | Search query text                 |
| `limit`      | `int`    | 10      | Max results (max 1000)            |
| `collection` | `string` | null    | Restrict to collection profile    |
| `hybrid`     | `bool`   | false   | Enable hybrid BM25 + vector search|
| `rerank`     | `bool`   | false   | Enable cross-encoder reranking    |
| `structural` | `bool`   | false   | Include structural embeddings     |

### `db.get` (Agent)

Fetch a single document by collection and key.

| Param        | Type     | Default  | Description     |
|--------------|----------|----------|-----------------|
| `collection` | `string` | required | Collection name |
| `key`        | `string` | required | Document `_key` |

### `db.list` (Agent)

List documents in a collection.

| Param        | Type     | Default | Description                |
|--------------|----------|---------|----------------------------|
| `collection` | `string` | null    | Collection name            |
| `limit`      | `int`    | 20      | Max results (max 1000)     |
| `paper`      | `string` | null    | Filter by paper_key        |

### `db.count` (Agent)

Get document count for a collection.

| Param        | Type     | Default  | Description     |
|--------------|----------|----------|-----------------|
| `collection` | `string` | required | Collection name |

### `db.check` (Agent)

Check a specific document's integrity.

| Param         | Type     | Default  | Description       |
|---------------|----------|----------|-------------------|
| `document_id` | `string` | required | Full `_id` string |

### `db.recent` (Agent)

Recently modified documents.

| Param   | Type  | Default | Description         |
|---------|-------|---------|---------------------|
| `limit` | `int` | 10      | Max results (max 1000) |

### `db.aql` (Admin)

Execute raw AQL query.

| Param   | Type     | Default  | Description           |
|---------|----------|----------|-----------------------|
| `aql`   | `string` | required | AQL query string      |
| `bind`  | `object` | `{}`     | Bind variables        |
| `limit` | `int`    | null     | Override result limit |

---

## Commands — Database Write

All write commands are **Admin** tier.

### `db.insert` (Admin)

Insert a document into a collection.

| Param        | Type     | Default  | Description       |
|--------------|----------|----------|-------------------|
| `collection` | `string` | required | Target collection |
| `data`       | `object` | required | Document body     |

### `db.update` (Admin)

Update fields on an existing document.

| Param        | Type     | Default  | Description     |
|--------------|----------|----------|-----------------|
| `collection` | `string` | required | Collection name |
| `key`        | `string` | required | Document `_key` |
| `data`       | `object` | required | Fields to merge |

### `db.delete` (Admin)

Delete a single document.

| Param        | Type     | Default  | Description     |
|--------------|----------|----------|-----------------|
| `collection` | `string` | required | Collection name |
| `key`        | `string` | required | Document `_key` |

### `db.purge` (Admin)

Delete a document and all connected edges.

| Param         | Type     | Default  | Description       |
|---------------|----------|----------|-------------------|
| `document_id` | `string` | required | Full `_id` string |

---

## Commands — Database Infrastructure

### `db.collections` (Internal)

List all collections in the database.

**Params:** `{}` (none)

### `db.stats` (Internal)

Database-level statistics.

**Params:** `{}` (none)

### `db.health` (Internal)

Database health check.

| Param     | Type   | Default | Description     |
|-----------|--------|---------|-----------------|
| `verbose` | `bool` | false   | Extended output |

### `db.create_collection` (Admin)

Create a new collection.

| Param             | Type     | Default | Description                          |
|-------------------|----------|---------|--------------------------------------|
| `name`            | `string` | required | Collection name                     |
| `collection_type` | `string` | null    | `"document"` (2) or `"edge"` (3)    |

### `db.create_index` (Admin)

Create a vector similarity index.

| Param        | Type     | Default    | Description          |
|--------------|----------|------------|----------------------|
| `collection` | `string` | required   | Collection name      |
| `dimension`  | `int`    | required   | Vector dimension     |
| `metric`     | `string` | `"cosine"` | Distance metric      |

---

## Commands — Graph

### `db.graph.traverse` (Agent)

Graph traversal from a starting vertex.

| Param       | Type     | Default      | Description                         |
|-------------|----------|--------------|-------------------------------------|
| `start`     | `string` | required     | Starting vertex `_id`               |
| `direction` | `string` | `"outbound"` | `outbound`, `inbound`, or `any`     |
| `min_depth` | `int`    | 1            | Minimum traversal depth             |
| `max_depth` | `int`    | 1            | Maximum traversal depth             |
| `limit`     | `int`    | null         | Max results (max 1000)              |
| `graph`     | `string` | null         | Named graph to traverse             |

### `db.graph.shortest_path` (Agent)

Find shortest path between two vertices.

| Param       | Type     | Default | Description           |
|-------------|----------|---------|-----------------------|
| `source`    | `string` | required | Source vertex `_id`  |
| `target`    | `string` | required | Target vertex `_id`  |
| `direction` | `string` | `"any"` | Direction constraint  |
| `graph`     | `string` | null    | Named graph           |

### `db.graph.neighbors` (Agent)

Direct neighbors of a vertex.

| Param       | Type     | Default | Description                     |
|-------------|----------|---------|---------------------------------|
| `vertex`    | `string` | required | Vertex `_id`                   |
| `direction` | `string` | `"any"` | `outbound`, `inbound`, or `any` |
| `limit`     | `int`    | null    | Max neighbors (max 1000)        |
| `graph`     | `string` | null    | Named graph                     |

### `db.graph.list` (Internal)

List named graphs.

**Params:** `{}` (none)

### `db.graph.create` (Admin)

Create a named graph via the Gharial API.

| Param              | Type     | Default | Description                              |
|--------------------|----------|---------|------------------------------------------|
| `name`             | `string` | required | Graph name                              |
| `edge_definitions` | `array`  | null    | Custom edge defs; falls back to schema   |

### `db.graph.drop` (Admin)

Drop a named graph.

| Param              | Type   | Default | Description                           |
|--------------------|--------|---------|---------------------------------------|
| `name`             | `string` | required | Graph name                          |
| `drop_collections` | `bool` | false   | Also drop edge/vertex collections     |
| `force`            | `bool` | false   | Confirmation flag                     |

### `db.graph.materialize` (Admin)

Materialize edges from implicit cross-reference fields. Reads edge
definitions from `hades_schema` (or NL statics fallback).

| Param      | Type     | Default | Description                                  |
|------------|----------|---------|----------------------------------------------|
| `edge`     | `string` | null    | Filter to single edge definition name        |
| `dry_run`  | `bool`   | false   | Preview mode — count without inserting        |
| `register` | `bool`   | false   | Also create named graphs via Gharial API      |

---

## Commands — Schema

Runtime ontology management. Schema definitions are stored per-database
in the `hades_schema` collection.

### `db.schema.init` (Admin)

Initialize or reset the `hades_schema` collection with a seed ontology.
Truncates existing schema before writing.

| Param  | Type     | Default  | Description                                     |
|--------|----------|----------|-------------------------------------------------|
| `seed` | `string` | required | `"nl"` (Nested Learning) or `"empty"` (blank)   |

### `db.schema.list` (Internal)

List all edge definitions and named graphs in the schema.

**Params:** `{}` (none)

### `db.schema.show` (Internal)

Show edge definition(s) or named graph by name. Returns all matching
edge definitions when multiple share a name (e.g., `nl_hecate_trace_edges`
has three definitions with different `source_field` values).

| Param  | Type     | Default  | Description                              |
|--------|----------|----------|------------------------------------------|
| `name` | `string` | required | Edge definition or named graph name      |

### `db.schema.version` (Internal)

Show schema metadata: version, checksum, relation count, feature dimension.

**Params:** `{}` (none)

---

## Commands — Embedding

### `embed.text` (Agent)

Generate embedding for a text string via the Jina V4 gRPC service.

| Param  | Type     | Default  | Description   |
|--------|----------|----------|---------------|
| `text` | `string` | required | Text to embed |

### `graph_embed.embed` (Agent)

Look up pre-computed structural embedding for a node.

| Param     | Type     | Default  | Description                   |
|-----------|----------|----------|-------------------------------|
| `node_id` | `string` | required | Node `_id` (`collection/key`) |

### `graph_embed.neighbors` (Agent)

Find k-nearest structural neighbors of a node.

| Param     | Type     | Default  | Description                   |
|-----------|----------|----------|-------------------------------|
| `node_id` | `string` | required | Node `_id` (`collection/key`) |
| `limit`   | `int`    | 10       | Number of neighbors           |

---

## Commands — Task Management

All task commands are **Agent** tier. They operate on the Persephone
kanban system in the configured database.

### `task.list` (Agent)

| Param       | Type     | Default  | Description                     |
|-------------|----------|----------|---------------------------------|
| `status`    | `string` | `"open"` | `open`, `closed`, or `all`      |
| `task_type` | `string` | null     | `task` or `epic`                |
| `parent`    | `string` | null     | Parent task key                 |
| `limit`     | `int`    | 50       | Max results                     |

### `task.show` (Agent)

| Param | Type     | Default  | Description |
|-------|----------|----------|-------------|
| `key` | `string` | required | Task `_key` |

### `task.create` (Agent)

| Param         | Type       | Default    | Description      |
|---------------|------------|------------|------------------|
| `title`       | `string`   | required   | Task title       |
| `description` | `string`   | null       | Task description |
| `task_type`   | `string`   | `"task"`   | `task` or `epic` |
| `parent`      | `string`   | null       | Parent task key  |
| `priority`    | `string`   | `"medium"` | Priority level   |
| `tags`        | `string[]` | `[]`       | Labels           |

### `task.update` (Agent)

| Param         | Type       | Default  | Description       |
|---------------|------------|----------|-------------------|
| `key`         | `string`   | required | Task `_key`       |
| `title`       | `string`   | null     | New title         |
| `description` | `string`   | null     | New description   |
| `priority`    | `string`   | null     | New priority      |
| `status`      | `string`   | null     | New status        |
| `add_tags`    | `string[]` | `[]`     | Tags to add       |
| `remove_tags` | `string[]` | `[]`     | Tags to remove    |

### `task.close` (Agent)

| Param     | Type     | Default  | Description     |
|-----------|----------|----------|-----------------|
| `key`     | `string` | required | Task `_key`     |
| `message` | `string` | null     | Closing message |

### `task.start` (Agent)

| Param | Type     | Default  | Description |
|-------|----------|----------|-------------|
| `key` | `string` | required | Task `_key` |

### `task.review` (Agent)

| Param     | Type     | Default  | Description      |
|-----------|----------|----------|------------------|
| `key`     | `string` | required | Task `_key`      |
| `message` | `string` | null     | Review message   |

### `task.approve` (Agent)

| Param   | Type     | Default  | Description               |
|---------|----------|----------|---------------------------|
| `key`   | `string` | required | Task `_key`               |
| `human` | `bool`   | false    | Human approval flag       |

### `task.block` (Agent)

| Param     | Type     | Default  | Description        |
|-----------|----------|----------|--------------------|
| `key`     | `string` | required | Task `_key`        |
| `message` | `string` | null     | Block reason       |
| `blocker` | `string` | null     | Blocking task key  |

### `task.unblock` (Agent)

| Param | Type     | Default  | Description |
|-------|----------|----------|-------------|
| `key` | `string` | required | Task `_key` |

### `task.handoff` (Agent)

| Param     | Type     | Default  | Description      |
|-----------|----------|----------|------------------|
| `key`     | `string` | required | Task `_key`      |
| `message` | `string` | null     | Handoff message  |

### `task.handoff_show` (Agent)

| Param | Type     | Default  | Description |
|-------|----------|----------|-------------|
| `key` | `string` | required | Task `_key` |

### `task.context` (Agent)

| Param | Type     | Default  | Description |
|-------|----------|----------|-------------|
| `key` | `string` | required | Task `_key` |

### `task.log` (Agent)

| Param   | Type     | Default  | Description |
|---------|----------|----------|-------------|
| `key`   | `string` | required | Task `_key` |
| `limit` | `int`    | null     | Max entries |

### `task.sessions` (Agent)

| Param | Type     | Default  | Description |
|-------|----------|----------|-------------|
| `key` | `string` | required | Task `_key` |

### `task.dep` (Agent)

| Param    | Type     | Default | Description              |
|----------|----------|---------|--------------------------|
| `key`    | `string` | required | Task `_key`             |
| `add`    | `string` | null    | Dependency key to add    |
| `remove` | `string` | null    | Dependency key to remove |
| `graph`  | `bool`   | false   | Show dependency graph    |

### `task.usage` (Agent)

**Params:** `{}` (none)

### `task.graph_integration` (Agent)

**Params:** `{}` (none)

---

## Commands — Code Quality

### `smell.check` (Agent)

Run code smell / compliance check on a file.

| Param     | Type     | Default  | Description               |
|-----------|----------|----------|---------------------------|
| `path`    | `string` | required | File path to check        |
| `verbose` | `bool`   | false    | Include detailed findings |

### `smell.verify` (Agent)

Verify specific compliance claims.

| Param    | Type       | Default  | Description            |
|----------|------------|----------|------------------------|
| `path`   | `string`   | required | File path to verify    |
| `claims` | `string[]` | `[]`     | Claims to verify       |

### `smell.report` (Agent)

Generate a compliance report.

| Param  | Type     | Default  | Description        |
|--------|----------|----------|--------------------|
| `path` | `string` | required | File path to check |

### `link_code_smell` (Agent)

Link a code smell to a source node.

| Param         | Type       | Default  | Description            |
|---------------|------------|----------|------------------------|
| `source_id`   | `string`   | required | Source node `_id`      |
| `smell_id`    | `string`   | required | Smell node `_id`       |
| `enforcement` | `string`   | required | Enforcement level      |
| `methods`     | `string[]` | `[]`     | Enforcement methods    |
| `summary`     | `string`   | null     | Summary text           |

---

## Commands — Codebase

### `codebase.stats` (Agent)

Codebase ingestion statistics.

**Params:** `{}` (none)

---

## Commands Not Exposed via Daemon

These commands are CLI-only. They are long-running batch operations,
service lifecycle management, or hardware queries that don't suit the
request-response socket model.

| Command                    | Reason                               |
|----------------------------|--------------------------------------|
| `ingest`                   | Batch pipeline, long-running         |
| `arxiv sync`               | Batch API fetch, long-running        |
| `codebase ingest`          | Batch AST parse, long-running        |
| `codebase update`          | Batch update, long-running           |
| `codebase validate`        | Batch validation                     |
| `graph-embed train`        | GPU training, minutes to hours       |
| `graph-embed update`       | Batch re-embed                       |
| `extract`                  | File I/O, not a query                |
| `link`                     | Interactive confirmation             |
| `db export`                | Batch export to file                 |
| `db backfill-text`         | Batch backfill                       |
| `db create-database`       | Infrastructure DDL (rare)            |
| `db databases`             | Multi-database query (rare)          |
| `db index-status`          | Index diagnostics (rare)             |
| `db search`                | Use `db.query` via daemon instead    |
| `embed service start/stop` | Service lifecycle management         |
| `embed gpu status/list`    | Hardware query                       |

## Connection Lifecycle

1. Client opens Unix stream to `/run/hades/hades.sock`.
2. Client writes a length-prefixed request.
3. Server reads the request, validates session and access tier.
4. Server dispatches to the handler, writes a length-prefixed response.
5. Steps 2-4 repeat for subsequent requests on the same connection.
6. Either side may close the connection at any time.

**Timeouts:**

| Timeout          | Duration | Scope          |
|------------------|----------|----------------|
| Idle timeout     | 30s      | Per-connection |
| Request timeout  | 60s      | Per-request    |
| Client default   | 10s      | Connect + read |

**Concurrency:**
- Each connection is serial: one request at a time, one response.
- The daemon accepts multiple concurrent connections (tokio task per connection).
- All connections share a single `ArangoPool`.

**Client reconnection:**
- Automatic single-retry on `BrokenPipe` or `ConnectionReset`.
- No retry for JSON errors, payload size violations, or other I/O errors.

**Socket lifecycle:**
- Daemon removes any stale socket file before binding (checks for live listener).
- Socket cleaned up on graceful shutdown (SIGTERM/SIGINT).
- Parent directory (`/run/hades/`) created if missing.

## Future Considerations

- **Pipelining:** `request_id` is reserved for a future version where
  multiple requests can be in-flight on a single connection.
- **Streaming:** Large result sets may benefit from chunked streaming
  responses. Not in v2.
- **Memory database bootstrap:** Agents may read their system prompt and
  memory context via daemon queries at session start (see
  `design-agent-memory-and-system-prompt.md`).
- **`--seed memory`:** A memory ontology seed for agent memory databases,
  following the same `hades_schema` pattern as `--seed nl`.
