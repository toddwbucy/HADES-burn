# HADES Daemon Protocol Specification

**Version:** 1.0-draft
**Status:** P6.1 deliverable
**Socket:** `/run/hades/hades.sock` (Unix domain socket)

## Overview

The HADES daemon exposes the CLI's query and CRUD surface over a persistent
Unix domain socket.  Local agents (weaver-tools, Persephone sessions) connect
once and issue commands without fork-per-call overhead.

**In scope:** fast, read-heavy operations and single-document writes.
**Out of scope:** batch ingest, training, arxiv sync, service lifecycle,
DDL (collection/database/index/graph creation and deletion).

## Wire Format

Length-prefixed JSON framing.  Every message (request and response) is:

```
[4 bytes: big-endian u32 payload length][JSON payload]
```

- **Maximum payload:** 16 MiB (16,777,216 bytes).  Messages exceeding this
  limit are rejected with error code `PAYLOAD_TOO_LARGE` without parsing.
- **Encoding:** UTF-8 JSON.  No trailing newline inside the payload.
- **Connection model:** one request, one response, serial per connection.
  Pipelining is reserved for a future version (see `request_id`).

### Example (hex)

```
00 00 00 1F {"command":"orient","params":{}}
```

The 4-byte header `0x0000001F` = 31, which is the byte length of the
JSON payload that follows.

## Request Schema

```jsonc
{
  // Optional, echoed in response.  Reserved for future pipelining.
  "request_id": "abc-123",

  // Required.  One of the command names defined below.
  "command": "db.query",

  // Required.  Command-specific parameters (may be empty object).
  "params": {
    "text": "attention mechanism",
    "limit": 10
  }
}
```

| Field        | Type              | Required | Description                               |
|--------------|-------------------|----------|-------------------------------------------|
| `request_id` | `string \| null`  | no       | Echoed in response; for client correlation |
| `command`    | `string`          | yes      | Command name (dotted namespace)            |
| `params`     | `object`          | yes      | Command-specific parameters                |

## Response Schema

```jsonc
{
  // Echoed from request (null if not provided).
  "request_id": "abc-123",

  // True if the command completed without error.
  "success": true,

  // Command-specific result payload (null on error).
  "data": { ... },

  // Human-readable error message (null on success).
  "error": null,

  // Machine-readable error code (null on success).
  "error_code": null
}
```

| Field        | Type             | Present  | Description                          |
|--------------|------------------|----------|--------------------------------------|
| `request_id` | `string \| null` | always   | Echoed from request                  |
| `success`    | `bool`           | always   | Whether the command succeeded         |
| `data`       | `object \| null` | always   | Result payload (null on error)        |
| `error`      | `string \| null` | always   | Error message (null on success)       |
| `error_code` | `string \| null` | always   | Machine-readable code (null on success) |

## Error Codes

| Code                  | Meaning                                             |
|-----------------------|-----------------------------------------------------|
| `UNKNOWN_COMMAND`     | `command` field not recognized                       |
| `INVALID_PARAMS`      | Missing or malformed `params` for the command        |
| `MALFORMED_JSON`      | Payload is not valid JSON                            |
| `PAYLOAD_TOO_LARGE`   | Payload exceeds 16 MiB limit                         |
| `NOT_FOUND`           | Requested document/collection/node does not exist    |
| `QUERY_FAILED`        | ArangoDB query execution error                       |
| `EMBEDDING_FAILED`    | Embedding service error                              |
| `INTERNAL`            | Unexpected server-side error                         |
| `READ_ONLY`           | Write attempted on read-only database                |

## Commands

Commands use dotted namespaces matching the CLI subcommand structure.
Each entry documents the `params` schema and the `data` response shape.

---

### `orient`

Metadata-first context orientation for a database.

**Params:**

| Field        | Type     | Default | Description               |
|--------------|----------|---------|---------------------------|
| `collection` | `string` | null    | Focus on specific collection |

**Response `data`:**
Matches current CLI JSON output: `{ database, profiles, total_collections,
total_documents, persephone }`.

**Maps to:** `Commands::Orient` (Python passthrough, then native)

---

### `status`

System status and workspace discovery.

**Params:**

| Field     | Type   | Default | Description     |
|-----------|--------|---------|-----------------|
| `verbose` | `bool` | false   | Extended output |

**Response `data`:**
Matches current CLI JSON output.

**Maps to:** `Commands::Status` (Python passthrough)

---

### `db.query`

Semantic search over documents (vector similarity + optional hybrid).

**Params:**

| Field        | Type     | Default   | Description                        |
|--------------|----------|-----------|------------------------------------|
| `text`       | `string` | required  | Search query text                  |
| `limit`      | `int`    | 10        | Max results                        |
| `collection` | `string` | null      | Restrict to collection profile     |
| `hybrid`     | `bool`   | false     | Enable hybrid BM25 + vector search |
| `rerank`     | `bool`   | false     | Enable reranking                   |
| `structural` | `bool`   | false     | Include structural embeddings      |

**Response `data`:**
```jsonc
{
  "results": [
    {
      "paper_key": "2501_12345",
      "title": "Attention Is All You Need",
      "text": "chunk text...",
      "score": 0.95,
      "chunk_index": 2,
      "total_chunks": 15
    }
  ],
  "count": 10
}
```

**Maps to:** `Commands::Db(DbCmd::Query { .. })`

---

### `db.aql`

Execute raw AQL query.

**Params:**

| Field   | Type     | Default  | Description                     |
|---------|----------|----------|---------------------------------|
| `aql`   | `string` | required | AQL query string                |
| `bind`  | `object` | `{}`     | Bind variables                  |
| `limit` | `int`    | null     | Override result limit           |

**Response `data`:**
```jsonc
{
  "results": [ ... ],
  "count": 42
}
```

**Maps to:** `Commands::Db(DbCmd::Aql { .. })`

---

### `db.get`

Fetch a single document by collection and key.

**Params:**

| Field        | Type     | Default  | Description     |
|--------------|----------|----------|-----------------|
| `collection` | `string` | required | Collection name |
| `key`        | `string` | required | Document `_key` |

**Response `data`:** The full document object.

**Maps to:** `Commands::Db(DbCmd::Get { .. })`

---

### `db.list`

List documents in a collection.

**Params:**

| Field        | Type     | Default  | Description                      |
|--------------|----------|----------|----------------------------------|
| `collection` | `string` | null     | Collection name                  |
| `limit`      | `int`    | 20       | Max results                      |
| `paper`      | `string` | null     | Filter by paper_key              |

**Response `data`:**
```jsonc
{
  "documents": [ ... ],
  "count": 20
}
```

**Maps to:** `Commands::Db(DbCmd::List { .. })`

---

### `db.insert`

Insert a document into a collection.

**Params:**

| Field        | Type     | Default  | Description          |
|--------------|----------|----------|----------------------|
| `collection` | `string` | required | Target collection    |
| `data`       | `object` | required | Document body        |

**Response `data`:**
```jsonc
{ "_key": "new_key", "_id": "collection/new_key", "_rev": "..." }
```

**Maps to:** `Commands::Db(DbCmd::Insert { .. })`

---

### `db.update`

Update fields on an existing document.

**Params:**

| Field        | Type     | Default  | Description          |
|--------------|----------|----------|----------------------|
| `collection` | `string` | required | Collection name      |
| `key`        | `string` | required | Document `_key`      |
| `data`       | `object` | required | Fields to merge      |

**Response `data`:**
```jsonc
{ "_key": "key", "_id": "collection/key", "_rev": "..." }
```

**Maps to:** `Commands::Db(DbCmd::Update { .. })`

---

### `db.delete`

Delete a single document.

**Params:**

| Field        | Type     | Default  | Description          |
|--------------|----------|----------|----------------------|
| `collection` | `string` | required | Collection name      |
| `key`        | `string` | required | Document `_key`      |

**Response `data`:**
```jsonc
{ "deleted": true, "_key": "key" }
```

**Maps to:** `Commands::Db(DbCmd::Delete { .. })`

---

### `db.count`

Get document count for a collection.

**Params:**

| Field        | Type     | Default  | Description     |
|--------------|----------|----------|-----------------|
| `collection` | `string` | required | Collection name |

**Response `data`:**
```jsonc
{ "collection": "hope_axioms", "count": 1234 }
```

**Maps to:** `Commands::Db(DbCmd::Count { .. })`

---

### `db.collections`

List all collections in the database.

**Params:** `{}` (none)

**Response `data`:**
```jsonc
{
  "collections": [
    { "name": "hope_axioms", "type": 2, "count": 1234 },
    { "name": "nl_links", "type": 3, "count": 5678 }
  ]
}
```

**Maps to:** `Commands::Db(DbCmd::Collections { .. })`

---

### `db.stats`

Database-level statistics.

**Params:** `{}` (none)

**Response `data`:** Matches current CLI JSON output.

**Maps to:** `Commands::Db(DbCmd::Stats { .. })`

---

### `db.health`

Database health check.

**Params:**

| Field     | Type   | Default | Description     |
|-----------|--------|---------|-----------------|
| `verbose` | `bool` | false   | Extended output |

**Response `data`:**
```jsonc
{ "status": "ok", "version": "3.12.x", "engine": "rocksdb" }
```

**Maps to:** `Commands::Db(DbCmd::Health { .. })`

---

### `db.check`

Check a specific document's integrity.

**Params:**

| Field         | Type     | Default  | Description       |
|---------------|----------|----------|-------------------|
| `document_id` | `string` | required | Full `_id` string |

**Response `data`:** Document integrity report.

**Maps to:** `Commands::Db(DbCmd::Check { .. })`

---

### `db.recent`

Recently modified documents.

**Params:**

| Field   | Type  | Default | Description |
|---------|-------|---------|-------------|
| `limit` | `int` | 10      | Max results |

**Response `data`:**
```jsonc
{ "documents": [ ... ], "count": 10 }
```

**Maps to:** `Commands::Db(DbCmd::Recent { .. })`

---

### `db.graph.traverse`

Graph traversal from a starting vertex.

**Params:**

| Field       | Type     | Default    | Description                     |
|-------------|----------|------------|---------------------------------|
| `start`     | `string` | required   | Starting vertex `_id`           |
| `direction` | `string` | `"outbound"` | `outbound`, `inbound`, or `any` |
| `min_depth` | `int`    | 1          | Minimum traversal depth         |
| `max_depth` | `int`    | 1          | Maximum traversal depth         |
| `graph`     | `string` | null       | Named graph to traverse         |

**Response `data`:**
```jsonc
{
  "vertices": [ ... ],
  "paths": [ ... ]
}
```

**Maps to:** `Commands::Db(DbCmd::Graph(DbGraphCmd::Traverse { .. }))`

---

### `db.graph.shortest_path`

Find shortest path between two vertices.

**Params:**

| Field    | Type     | Default  | Description          |
|----------|----------|----------|----------------------|
| `source` | `string` | required | Source vertex `_id`   |
| `target` | `string` | required | Target vertex `_id`   |
| `graph`  | `string` | null     | Named graph           |

**Response `data`:**
```jsonc
{
  "vertices": [ ... ],
  "edges": [ ... ],
  "length": 3
}
```

**Maps to:** `Commands::Db(DbCmd::Graph(DbGraphCmd::ShortestPath { .. }))`

---

### `db.graph.neighbors`

Direct neighbors of a vertex.

**Params:**

| Field       | Type     | Default    | Description                     |
|-------------|----------|------------|---------------------------------|
| `vertex`    | `string` | required   | Vertex `_id`                    |
| `direction` | `string` | `"any"`    | `outbound`, `inbound`, or `any` |
| `limit`     | `int`    | 100        | Max neighbors                   |

**Response `data`:**
```jsonc
{
  "neighbors": [
    { "_id": "...", "_key": "...", "title": "..." }
  ],
  "count": 42
}
```

**Maps to:** `Commands::Db(DbCmd::Graph(DbGraphCmd::Neighbors { .. }))`

---

### `db.graph.list`

List named graphs.

**Params:** `{}` (none)

**Response `data`:**
```jsonc
{ "graphs": [ { "name": "nl_core", "edge_definitions": [ ... ] } ] }
```

**Maps to:** `Commands::Db(DbCmd::Graph(DbGraphCmd::List { .. }))`

---

### `embed.text`

Generate embedding for a text string.

**Params:**

| Field  | Type     | Default  | Description        |
|--------|----------|----------|--------------------|
| `text` | `string` | required | Text to embed      |

**Response `data`:**
```jsonc
{
  "embedding": [0.123, -0.456, ...],
  "dimension": 2048,
  "model": "jina-embeddings-v4"
}
```

**Maps to:** `Commands::Embed(EmbedCmd::Text { .. })`

---

### `graph_embed.embed`

Look up pre-computed structural embedding for a node.

**Params:**

| Field     | Type     | Default  | Description                      |
|-----------|----------|----------|----------------------------------|
| `node_id` | `string` | required | Node `_id` (`collection/key`)    |

**Response `data`:**
```jsonc
{
  "node_id": "hope_axioms/ax_001",
  "label": "Axiom of Choice",
  "embedding_dim": 128,
  "embedding": [0.12, -0.34, ...]
}
```

**Maps to:** `Commands::GraphEmbed(GraphEmbedCmd::Embed { .. })`

---

### `graph_embed.neighbors`

Find k-nearest structural neighbors of a node.

**Params:**

| Field     | Type     | Default  | Description                     |
|-----------|----------|----------|---------------------------------|
| `node_id` | `string` | required | Node `_id` (`collection/key`)   |
| `limit`   | `int`    | 10       | Number of neighbors             |

**Response `data`:**
```jsonc
{
  "query_node": "hope_axioms/ax_001",
  "k": 10,
  "neighbors": [
    { "id": "...", "label": "...", "collection": "...", "similarity": 0.89 }
  ]
}
```

**Maps to:** `Commands::GraphEmbed(GraphEmbedCmd::Neighbors { .. })`

---

### `task.list`

List Persephone tasks.

**Params:**

| Field    | Type     | Default  | Description                    |
|----------|----------|----------|--------------------------------|
| `status` | `string` | `"open"` | Filter: `open`, `closed`, `all` |
| `type`   | `string` | null     | Filter: `task`, `epic`          |
| `parent` | `string` | null     | Parent task key                 |
| `limit`  | `int`    | 50       | Max results                     |

**Response `data`:**
```jsonc
{ "tasks": [ ... ], "count": 12 }
```

**Maps to:** `Commands::Task(TaskCmd::List { .. })`

---

### `task.show`

Get details of a single task.

**Params:**

| Field | Type     | Default  | Description |
|-------|----------|----------|-------------|
| `key` | `string` | required | Task `_key` |

**Response `data`:** Full task object.

**Maps to:** `Commands::Task(TaskCmd::Show { .. })`

---

### `task.create`

Create a new task.

**Params:**

| Field         | Type       | Default    | Description        |
|---------------|------------|------------|--------------------|
| `title`       | `string`   | required   | Task title         |
| `description` | `string`   | null       | Task description   |
| `type`        | `string`   | `"task"`   | `task` or `epic`   |
| `parent`      | `string`   | null       | Parent task key    |
| `priority`    | `string`   | `"medium"` | Priority level     |
| `tags`        | `string[]` | `[]`       | Labels             |

**Response `data`:** Created task object.

**Maps to:** `Commands::Task(TaskCmd::Create { .. })`

---

### `task.update`

Update task fields.

**Params:**

| Field         | Type       | Default  | Description         |
|---------------|------------|----------|---------------------|
| `key`         | `string`   | required | Task `_key`         |
| `title`       | `string`   | null     | New title           |
| `description` | `string`   | null     | New description     |
| `priority`    | `string`   | null     | New priority        |
| `add_tags`    | `string[]` | `[]`     | Tags to add         |
| `remove_tags` | `string[]` | `[]`     | Tags to remove      |

**Response `data`:** Updated task object.

**Maps to:** `Commands::Task(TaskCmd::Update { .. })`

---

### `task.close`

Close a task.

**Params:**

| Field     | Type     | Default  | Description        |
|-----------|----------|----------|--------------------|
| `key`     | `string` | required | Task `_key`        |
| `message` | `string` | null     | Closing message    |

**Response `data`:** Updated task object with `status: "closed"`.

**Maps to:** `Commands::Task(TaskCmd::Close { .. })`

---

### `task.context`

Get task context (dependencies, history, related).

**Params:**

| Field | Type     | Default  | Description |
|-------|----------|----------|-------------|
| `key` | `string` | required | Task `_key` |

**Response `data`:** Context object (task + dependencies + logs).

**Maps to:** `Commands::Task(TaskCmd::Context { .. })`

---

### `smell.check`

Run code smell / compliance check on a file.

**Params:**

| Field     | Type     | Default  | Description               |
|-----------|----------|----------|---------------------------|
| `path`    | `string` | required | File path to check        |
| `verbose` | `bool`   | false    | Include detailed findings |

**Response `data`:**
```jsonc
{
  "path": "/path/to/file.py",
  "smells": [ { "code": "CS-01", "message": "...", "line": 42 } ],
  "clean": false
}
```

**Maps to:** `Commands::Smell(SmellCmd::Check { .. })`

---

## Commands Excluded from Daemon

These commands are **intentionally not exposed** over the socket:

| Command                | Reason                                  |
|------------------------|-----------------------------------------|
| `ingest`               | Batch pipeline, long-running            |
| `arxiv sync`           | Batch API fetch, long-running           |
| `codebase ingest`      | Batch AST parse, long-running           |
| `codebase update`      | Batch update, long-running              |
| `graph-embed train`    | GPU training, minutes to hours          |
| `graph-embed update`   | Batch re-embed                          |
| `extract`              | File I/O, not a query                   |
| `link`                 | Interactive confirmation                |
| `db purge`             | Destructive bulk delete                 |
| `db create`            | DDL (collection creation)               |
| `db create-database`   | DDL (database creation)                 |
| `db create-index`      | DDL (index creation)                    |
| `db graph create/drop` | DDL (graph definition)                  |
| `db graph materialize` | Batch materialization                   |
| `db export`            | Batch export to file                    |
| `db backfill-text`     | Batch backfill                          |
| `embed service *`      | Service lifecycle management            |
| `embed gpu *`          | Hardware query (not frequently needed)  |

## Connection Lifecycle

1. Client opens Unix stream to `/run/hades/hades.sock`.
2. Client writes a length-prefixed request.
3. Server reads the request, dispatches to the handler, writes a
   length-prefixed response.
4. Client reads the response.
5. Steps 2-4 repeat for subsequent requests on the same connection.
6. Either side may close the connection at any time.

**Timeouts:**
- Server: 30s idle timeout per connection (no request received).
- Server: 60s per-request execution timeout.
- Client: should set a read timeout appropriate to the expected
  command latency (recommend 10s for queries, 30s for writes).

**Concurrency:**
- Each connection is serial: one request at a time, one response.
- The daemon accepts multiple concurrent connections.
- All connections share a single `ArangoPool`.
- No cross-connection state or sessions.

## Shared State

The daemon holds:
- `ArangoPool` — persistent ArangoDB connection pool (reader + writer sockets).
- `HadesConfig` — loaded once at startup.
- No per-connection state, no caches (ArangoDB is the source of truth).

## Future Considerations

- **Pipelining:** `request_id` is reserved for a future version where
  multiple requests can be in-flight on a single connection.
- **Streaming:** Large result sets (e.g., traversals) may benefit from
  chunked streaming responses.  Not in v1.
- **Authentication:** v1 relies on Unix socket file permissions
  (owner/group).  Future versions may add token-based auth.
- **Notifications:** Server-push events (e.g., task status changes)
  are out of scope for v1.
