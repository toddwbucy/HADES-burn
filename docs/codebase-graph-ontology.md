# Codebase Graph Ontology Specification

**Status:** v2.0 — implemented
**Date:** 2026-04-06
**Scope:** ArangoDB schema for `codebase ingest` — collections, document contracts, edge definitions, named graph, indices
**Relates to:** task_149770 (Codebase schema ontology)

---

## 1. Executive Summary

The codebase graph stores the structural representation of ingested source code: files, symbols, text chunks, embedding vectors, and the relationships between them. This document defines the **canonical schema** — the contracts that all writers (ingest pipeline, rust-analyzer enrichment, import resolution) must obey, and all readers (search, traversal, GraphSAGE training) can rely on.

### Design Principles

1. **Schema constrains the agent.** The ontology defines what can exist. Code populates it; nothing invents new structure at runtime.
2. **Primitives first.** Five graph semantic primitives — `file`, `module`, `type`, `callable`, `value` — are the only vertex kinds the graph reasons in. Everything else is either an edge or metadata on a primitive. If it's not a primitive, it doesn't get a vertex.
3. **Collection-per-relation.** Each edge type is its own ArangoDB edge collection with specific `_from`/`_to` vertex constraints. The collection name IS the relation type — no discriminator fields, no translation layer for GNN training.
4. **One canonical document shape per collection.** Optional fields are allowed; polymorphic documents are not. Every field is documented with its type, requiredness, and provenance.
5. **Marker attributes, not marker edges.** Boolean properties (PyO3, FFI, unsafe) belong on the symbol document, not as self-referential edges with zero traversal value.
6. **Indices match query patterns.** Every field used in a `FILTER` or `JOIN` gets a persistent index.

---

## 2. Named Graph Definition

```text
Graph: "codebase_graph"

Edge collections:
  codebase_defines_edges
    _from: [codebase_files]
    _to:   [codebase_symbols]

  codebase_calls_edges
    _from: [codebase_symbols]
    _to:   [codebase_symbols]

  codebase_implements_edges
    _from: [codebase_symbols]
    _to:   [codebase_symbols]

  codebase_imports_edges
    _from: [codebase_files]
    _to:   [codebase_files, codebase_symbols]

Orphan collections: [codebase_chunks, codebase_embeddings]
```

**Why a named graph?** ArangoDB's graph traversal API (`FOR v, e IN 1..N OUTBOUND/INBOUND`) requires either a named graph or explicit edge collection lists. A named graph also enforces that `_from`/`_to` point only to declared vertex collections — preventing stale references to deleted documents or cross-ontology leaks (e.g., an edge accidentally pointing into `arxiv_metadata`).

**Why separate edge collections?** Each collection carries its own `_from`/`_to` constraint. ArangoDB rejects a `codebase_defines_edges` document with `_from` pointing to `codebase_symbols` — the graph enforces directionality at insert time. Separate collections also align with the NL graph convention (22 edge collections, collection name = RGCN relation type) and enable collection-level skipping during random walks in `hades-prefetch`.

**Why orphan collections?** Chunks and embeddings are not vertices in the traversal graph. They are accessed by key lookup from file/symbol context, not by edge traversal. Including them as vertices would bloat traversal queries with irrelevant hops.

### Gharial Payload

```json
{
  "name": "codebase_graph",
  "edgeDefinitions": [
    {
      "collection": "codebase_defines_edges",
      "from": ["codebase_files"],
      "to": ["codebase_symbols"]
    },
    {
      "collection": "codebase_calls_edges",
      "from": ["codebase_symbols"],
      "to": ["codebase_symbols"]
    },
    {
      "collection": "codebase_implements_edges",
      "from": ["codebase_symbols"],
      "to": ["codebase_symbols"]
    },
    {
      "collection": "codebase_imports_edges",
      "from": ["codebase_files"],
      "to": ["codebase_files", "codebase_symbols"]
    }
  ],
  "orphanCollections": ["codebase_chunks", "codebase_embeddings"]
}
```

---

## 3. Graph Semantic Primitives

Five primitives. Every vertex in the codebase graph is exactly one of these. Every GNN node feature, every traversal filter, every chunking boundary decision operates on this taxonomy.

| Primitive | Stored in | What it represents |
|-----------|-----------|-------------------|
| `file` | `codebase_files` | A source file — the top-level container |
| `module` | `codebase_symbols` | A namespace container (Rust `mod`, Python module/package) |
| `type` | `codebase_symbols` | A type definition (struct, enum, trait, class, type alias) |
| `callable` | `codebase_symbols` | Something that can be invoked (function, method, macro) |
| `value` | `codebase_symbols` | A named binding (constant, variable, static) |

### What is NOT a primitive

| Concept | Why not | Where it lives instead |
|---------|---------|----------------------|
| `import` | A reference to a primitive, not a primitive itself | `codebase_imports_edges` — the edge IS the import |
| `impl` block | Scaffolding that binds callables to a type, not an identity | Methods extracted individually as `callable`; `parent_symbol` field points to the type |

### Mapping from language-specific kinds

The `lang_kind` field on symbol documents preserves the original AST classification. The `kind` field always holds the universal primitive.

| `lang_kind` | Primitive (`kind`) | Languages |
|-------------|-------------------|-----------|
| `function` | `callable` | Rust, Python |
| `method` | `callable` | Python |
| `macro` | `callable` | Rust |
| `struct` | `type` | Rust |
| `enum` | `type` | Rust |
| `trait` | `type` | Rust |
| `class` | `type` | Python |
| `type_alias` | `type` | Rust |
| `constant` | `value` | Rust, Python |
| `variable` | `value` | Python |
| `module` | `module` | Rust, Python |

Chunking happens at `callable` and `type` boundaries universally — not "function in Rust" or "def in Python." The chunker sees primitives, not language artifacts.

---

## 4. Vertex Collections

### 4.1 `codebase_files` — File Metadata

One document per ingested source file. Primitive kind: `file`. The primary vertex for `defines` and `imports` edges.

| Field | Type | Required | Source | Description |
|-------|------|----------|--------|-------------|
| `_key` | string | yes | `keys::file_key(rel_path)` | Normalized path: `.` and `/` replaced with `_` |
| `path` | string | yes | ingest | Relative path from project root (e.g., `src/lib.rs`) |
| `language` | string | yes | ingest | Language identifier: `"rust"`, `"python"`, `"go"`, `"typescript"`, `"javascript"`, `"java"`, `"c"`, `"cpp"` |
| `kind` | string | yes | ingest | Always `"file"` — the graph semantic primitive |
| `status` | string | yes | ingest | Processing status: `"PROCESSED"` |
| `ingested_at` | string | yes | ingest | ISO 8601 timestamp of last ingest |
| `symbol_hash` | string | yes | ingest | SHA-256 hex digest of sorted symbol names — used for incremental skip |
| `symbol_count` | integer | yes | ingest | Number of symbols extracted from this file |
| `chunk_count` | integer | yes | ingest | Number of text chunks produced |
| `embedding_count` | integer | yes | ingest | Number of embeddings stored (0 if embedder unavailable) |
| `total_lines` | integer | yes | ingest | Total line count |
| `metrics` | object | yes | ingest | See **Metrics Object** below |
| `ra_analyzed` | boolean | no | rust-analyzer | `true` if rust-analyzer enrichment completed |
| `ra_symbol_count` | integer | no | rust-analyzer | Symbol count from rust-analyzer (may differ from syn count) |
| `ra_analyzed_at` | string | no | rust-analyzer | ISO 8601 timestamp of rust-analyzer pass |

#### Metrics Object

Nested under `metrics` on every file document.

| Field | Type | Description |
|-------|------|-------------|
| `total_lines` | integer | Total lines in file |
| `lines_of_code` | integer | Non-blank, non-comment lines |
| `blank_lines` | integer | Blank lines |
| `comment_lines` | integer | Comment lines |
| `cyclomatic_complexity` | integer | McCabe cyclomatic complexity |
| `max_nesting_depth` | integer | Maximum block nesting depth |

#### Key Generation

```text
file_key("src/lib.rs")           → "src_lib_rs"
file_key("core/models.py")      → "core_models_py"
file_key("tests/test_config.rs") → "tests_test_config_rs"
```

Rule: replace all `.` and `/` characters with `_`. No version stripping. No hashing — keys are human-readable.

---

### 4.2 `codebase_symbols` — Symbol Metadata

One document per code symbol. Only primitives get symbol documents: `module`, `type`, `callable`, `value`. Import statements and impl blocks are not symbols — they produce edges and metadata respectively.

| Field | Type | Required | Source | Description |
|-------|------|----------|--------|-------------|
| `_key` | string | yes | `keys::symbol_key(file_key, qualified_name)` | Deterministic: `{file_key}__{readable}__{hash8}` |
| `name` | string | yes | all | Bare symbol name (e.g., `new`, `Config`, `MyClass`) |
| `qualified_name` | string | yes | all | Fully qualified name (e.g., `Config::new`, `MyModule.MyClass`) |
| `kind` | string | yes | all | Graph semantic primitive: `"module"`, `"type"`, `"callable"`, `"value"` |
| `lang_kind` | string | yes | all | Original language-specific kind: `"function"`, `"struct"`, `"class"`, etc. |
| `file_key` | string | yes | all | References parent `codebase_files._key` |
| `file_path` | string | yes | all | Relative path (denormalized for query convenience) |
| `start_line` | integer | yes | all | 1-indexed start line |
| `end_line` | integer | yes | all | 1-indexed end line (inclusive) |
| `visibility` | string | no | rust-analyzer, syn | `"pub"`, `"pub(crate)"`, `"pub(super)"`, `"private"` |
| `signature` | string | no | rust-analyzer | Full type signature (e.g., `fn new(config: &Config) -> Self`) |
| `parent_symbol` | string | no | rust-analyzer | Enclosing symbol's qualified name (e.g., `Config` for `Config::new`) |
| `impl_trait` | string | no | rust-analyzer | Trait being implemented (e.g., `Display` for `impl Display for Config`) |
| `is_pyo3` | boolean | no | rust-analyzer | `true` if exposed to Python via `#[pyfunction]`/`#[pymethods]` |
| `is_ffi` | boolean | no | rust-analyzer | `true` if exposed via `extern "C"` or `#[no_mangle]` |
| `is_unsafe` | boolean | no | rust-analyzer | `true` if declared `unsafe` |
| `is_async` | boolean | no | all | `true` if declared `async` |
| `derives` | string[] | no | rust-analyzer | Derive macro names (e.g., `["Debug", "Clone", "Serialize"]`) |
| `python_name` | string | no | rust-analyzer | PyO3 `#[pyo3(name = "...")]` override |
| `decorators` | string[] | no | python | Python decorator names (e.g., `["staticmethod", "property"]`) |
| `docstring` | string | no | python | First line of docstring |
| `bases` | string[] | no | python | Base classes (e.g., `["BaseModel", "ABC"]`) |
| `parameters` | string[] | no | python | Function parameter names |
| `source` | string | no | all | Extraction provenance: `"syn"`, `"rust-analyzer"`, `"python-ast"` |
| `analyzed_at` | string | no | all | ISO 8601 timestamp of extraction |

#### Key Generation

```text
symbol_key("src_lib_rs", "Config::new")
  → readable: "Config__new"
  → hash8:    SHA-256("Config::new")[:8]
  → result:   "src_lib_rs__Config__new__a1b2c3d4"
```

The hash suffix prevents collisions from lossy normalization (e.g., `Vec<T>` and `Vec_T_` would produce the same readable prefix but different hashes).

#### Enrichment Protocol

Symbols may be written twice during a single ingest run:

1. **First pass (syn/AST):** Creates the document with `name`, `qualified_name`, `kind`, `lang_kind`, `file_key`, `file_path`, `start_line`, `end_line`, and language-specific metadata. Sets `source: "syn"` or `source: "python-ast"`. Syn produces qualified names from AST context — methods inside `impl Foo` get `qualified_name: "Foo::bar"`, not bare `"bar"`.

2. **Second pass (rust-analyzer):** Overwrites with the full document including `signature`, `parent_symbol`, `impl_trait`, PyO3/FFI flags, derives. Sets `source: "rust-analyzer"`.

Both passes use the same `qualified_name` → same `_key`. The second pass uses `overwrite: true` (ArangoDB replace semantics). This is safe because rust-analyzer output is strictly a superset of syn output for the same symbol.

---

### 4.3 `codebase_chunks` — Text Chunks

One document per text chunk. Chunks are contiguous spans of source code text, produced by the AST-aware chunking algorithm. Chunk boundaries align to `callable` and `type` primitives — the chunker splits at primitive boundaries, not language-specific constructs.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `_key` | string | yes | `{file_key}_chunk_{index}` |
| `file_key` | string | yes | References parent `codebase_files._key` |
| `chunk_index` | integer | yes | 0-indexed position within the file |
| `total_chunks` | integer | yes | Total chunk count for this file |
| `text` | string | yes | The chunk text content |
| `start_char` | integer | yes | Byte offset from start of file |
| `end_char` | integer | yes | Byte offset of chunk end |
| `symbols` | string[] | yes | Symbol keys (`codebase_symbols._key`) whose span overlaps this chunk |

The `symbols` field is populated during ingest via interval intersection in Rust. Both symbol spans (from syn/AST) and chunk byte offsets are in memory concurrently — the intersection is a single O(symbols + chunks) sorted merge, microseconds per file.

Empty array `[]` is valid for module-level gap chunks (inter-definition code, trailing content) that contain no named symbol definitions.

Rust-analyzer is a post-pass that enriches symbol documents only. Chunk documents reference syn-extracted symbol keys, which are deterministic and stable. The keys are identical across both passes because both derive from the same `qualified_name`.

#### Key Generation

```text
chunk_key("src_lib_rs", 0)  → "src_lib_rs_chunk_0"
chunk_key("src_lib_rs", 3)  → "src_lib_rs_chunk_3"
```

---

### 4.4 `codebase_embeddings` — Embedding Vectors

One document per embedding vector. Each embedding corresponds to one chunk.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `_key` | string | yes | `{chunk_key}_emb` |
| `chunk_key` | string | yes | References parent `codebase_chunks._key` |
| `file_key` | string | yes | References grandparent `codebase_files._key` (denormalized) |
| `embedding` | float[] | yes | Vector of dimension `dimension` |
| `model` | string | yes | Model identifier (e.g., `"jina-embeddings-v3"`) |
| `model_hash` | string | yes | Hash of model weights — used to detect stale embeddings after model changes |
| `dimension` | integer | yes | Vector dimension (e.g., `2048`) |

#### Key Generation

```text
embedding_key("src_lib_rs_chunk_0")  → "src_lib_rs_chunk_0_emb"
```

---

## 5. Edge Collections

Four edge collections, one per relation type. The collection name IS the relation — no `type` discriminator field on edge documents.

### Common Fields (all edge collections)

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `_key` | string | yes | Deterministic: `keys::edge_key(from, kind, to)` |
| `_from` | string | yes | Source vertex ID |
| `_to` | string | yes | Target vertex ID |

#### Edge Key Generation

```text
edge_key("src_lib_rs", "defines", "src_lib_rs__Config__new__e5f6a7b8")
  → from_prefix: "src_lib_rs" (first 20 chars)
  → to_prefix:   "src_lib_rs__Config__" (first 20 chars)
  → hash8:       SHA-256("src_lib_rs|defines|src_lib_rs__Config__new__e5f6a7b8")[:8]
  → result:      "src_lib_rs__________defines__src_lib_rs__Config____a1b2c3d4"
```

Deterministic keys enable idempotent re-runs — re-ingesting the same codebase produces the same edge keys, allowing `overwrite: true` without duplication. The `kind` parameter remains in `edge_key()` to ensure the same `(from, to)` pair in different collections produces different keys.

---

### 5.1 `codebase_defines_edges`

**Semantics:** A file defines (contains the declaration of) a symbol.

| Constraint | Value |
|-----------|-------|
| `_from` | `codebase_files/{file_key}` |
| `_to` | `codebase_symbols/{symbol_key}` |
| Cardinality | 1 file : N symbols (one edge per symbol) |

| Metadata Field | Type | Required | Description |
|---------------|------|----------|-------------|
| `file_path` | string | yes | Relative path (denormalized) |
| `symbol_name` | string | yes | Bare symbol name |

**Example:**
```json
{
  "_key": "src_lib_rs__________defines__src_lib_rs__Config____a1b2c3d4",
  "_from": "codebase_files/src_lib_rs",
  "_to": "codebase_symbols/src_lib_rs__Config__new__e5f6a7b8",
  "file_path": "src/lib.rs",
  "symbol_name": "new"
}
```

---

### 5.2 `codebase_calls_edges`

**Semantics:** A callable invokes another callable.

| Constraint | Value |
|-----------|-------|
| `_from` | `codebase_symbols/{caller_key}` |
| `_to` | `codebase_symbols/{callee_key}` |
| Cardinality | N:M |

| Metadata Field | Type | Required | Description |
|---------------|------|----------|-------------|
| `caller` | string | yes | Caller's qualified name |
| `callee` | string | yes | Callee's qualified name |

---

### 5.3 `codebase_implements_edges`

**Semantics:** A callable implements a trait method or interface.

| Constraint | Value |
|-----------|-------|
| `_from` | `codebase_symbols/{impl_method_key}` |
| `_to` | `codebase_symbols/{trait_method_key}` |
| Cardinality | N:M |

| Metadata Field | Type | Required | Description |
|---------------|------|----------|-------------|
| `implementor` | string | yes | Implementing type's qualified name |
| `trait_name` | string | yes | Trait's qualified name |

---

### 5.4 `codebase_imports_edges`

**Semantics:** A file imports a symbol (or module) from another file. This is where import statements live — imports are edges, not vertices.

| Constraint | Value |
|-----------|-------|
| `_from` | `codebase_files/{importing_file_key}` |
| `_to` | `codebase_symbols/{imported_symbol_key}` **or** `codebase_files/{imported_file_key}` |
| Cardinality | N:M |

The `_to` target depends on resolution success:
- **Resolved:** Points to the specific symbol in `codebase_symbols`
- **Unresolved:** Falls back to the file in `codebase_files` (module-level import)

| Metadata Field | Type | Required | Description |
|---------------|------|----------|-------------|
| `source_path` | string | yes | Importing file's relative path |
| `target_path` | string | yes | Target file's relative path |
| `symbol_name` | string | no | Imported symbol name (absent for module-level) |
| `use_path` | string | no | Rust: full use path (e.g., `crate::config::Config`) |
| `module_path` | string | no | Python: module path (e.g., `core.config`) |
| `resolved` | boolean | yes | Whether the import resolved to a specific symbol |
| `style` | string | no | Import style: `"use"`, `"from_import"`, `"import"`, `"wildcard"` |

---

### 5.5 Retired Edge Types

The following edge types from the initial implementation are **removed from the ontology** and replaced by symbol document attributes:

| Former Edge Type | Replacement | Rationale |
|-----------------|-------------|-----------|
| `pyo3_exposes` | `codebase_symbols.is_pyo3: true` | Self-referential edge (`_from == _to`) has zero traversal value. The information is a property of the symbol, not a relationship between symbols. |
| `ffi_exposes` | `codebase_symbols.is_ffi: true` | Same rationale. |

**Migration path:** If cross-language bridging is needed later (e.g., Rust symbol → Python symbol mapping), a new edge collection `codebase_bridges_edges` should be introduced with `_from` in the Rust symbol and `_to` in a Python-side symbol — a real relationship with traversal value.

---

## 6. Index Definitions

### 6.1 Persistent Indices

These indices support the known query patterns in search, traversal, and incremental ingest.

| Collection | Fields | Type | Justification |
|-----------|--------|------|---------------|
| `codebase_chunks` | `[file_key]` | persistent | Fetch all chunks for a file, cascade delete |
| `codebase_chunks` | `[symbols[*]]` | persistent (array) | Reverse lookup: find chunks containing a specific symbol |
| `codebase_embeddings` | `[file_key]` | persistent | Delete stale embeddings by file, embedding backfill check |
| `codebase_embeddings` | `[chunk_key]` | persistent | Join embedding to its chunk (vector search phase 3) |
| `codebase_symbols` | `[file_key]` | persistent | Fetch all symbols for a file, symbol count verification |
| `codebase_symbols` | `[kind]` | persistent | Query by primitive kind (5 values: module, type, callable, value) |

### 6.2 Automatic Indices (ArangoDB-managed)

These are created automatically and do not need explicit creation:

| Collection | Fields | Type | Notes |
|-----------|--------|------|-------|
| All collections | `[_key]` | primary | Always present |
| `codebase_defines_edges` | `[_from]`, `[_to]` | edge index | Automatic for edge collections |
| `codebase_calls_edges` | `[_from]`, `[_to]` | edge index | Automatic for edge collections |
| `codebase_implements_edges` | `[_from]`, `[_to]` | edge index | Automatic for edge collections |
| `codebase_imports_edges` | `[_from]`, `[_to]` | edge index | Automatic for edge collections |

### 6.3 Index Creation

```javascript
// Chunks: lookup by parent file
db.codebase_chunks.ensureIndex({ type: "persistent", fields: ["file_key"], name: "idx_chunks_file_key" });
// Chunks: reverse symbol lookup (array index)
db.codebase_chunks.ensureIndex({ type: "persistent", fields: ["symbols[*]"], name: "idx_chunks_symbols" });

// Embeddings: lookup by parent file and parent chunk
db.codebase_embeddings.ensureIndex({ type: "persistent", fields: ["file_key"], name: "idx_embeddings_file_key" });
db.codebase_embeddings.ensureIndex({ type: "persistent", fields: ["chunk_key"], name: "idx_embeddings_chunk_key" });

// Symbols: lookup by parent file and by primitive kind
db.codebase_symbols.ensureIndex({ type: "persistent", fields: ["file_key"], name: "idx_symbols_file_key" });
db.codebase_symbols.ensureIndex({ type: "persistent", fields: ["kind"], name: "idx_symbols_kind" });
```

---

## 7. Collection Lifecycle

### 7.1 Creation

Collections, graph, and indices are created by `ensure_collections()` at the start of each `codebase ingest` run. Creation is idempotent — existing collections, graphs, and indices are skipped.

Creation order matters: vertex collections must exist before the named graph is created, because the graph definition references them.

```text
1. Create document collections: files, chunks, embeddings, symbols     (4)
2. Create edge collections: defines, calls, implements, imports        (4)
3. Create named graph: codebase_graph (via POST /_api/gharial)         (1)
4. Ensure persistent indices on document collections                   (6)
```

### 7.2 Incremental Ingest

Documents use `overwrite: true` (ArangoDB `REPLACE` semantics). This means:
- Re-ingesting the same file replaces its document entirely
- Stale embeddings are explicitly deleted before new ones are written
- Edge documents with the same `_key` (deterministic from `edge_key()`) are replaced
- New symbols overwrite old symbols for the same key

### 7.3 Deletion Cascade

When a file is removed from the codebase and re-ingested, orphaned documents remain. A future `codebase purge` command should:

1. Identify `codebase_files` documents whose `path` no longer exists on disk
2. Delete all `codebase_chunks` with matching `file_key`
3. Delete all `codebase_embeddings` with matching `file_key`
4. Delete all `codebase_symbols` with matching `file_key`
5. Delete from all 4 edge collections where `_from` or `_to` references the deleted file or its symbols
6. Delete the `codebase_files` document

This cascade requires the `file_key` indices defined in Section 6.

---

## 8. Relationship to NL Graph

The codebase graph and the NL knowledge graph are **separate named graphs** in the same database. They share no collections and no edges.

Future bridging between them (e.g., linking a code symbol to the equation it implements) would use a **new edge collection** — not by mixing vertices across graph boundaries:

```text
Potential future bridge:
  Edge collection: nl_code_equation_edges
  _from: codebase_symbols (code callable)
  _to:   {paper}_equations (mathematical equation)

  This is already reserved as EDGE_COLLECTION_NAMES[5]
  in graph/schema.rs: "nl_code_equation_edges"
```

This bridge is explicitly **out of scope** for the codebase ontology. It belongs to the NL graph schema and would be defined there when the time comes.

---

## 9. RGCN Training Considerations

The NL graph uses `EDGE_COLLECTION_NAMES` (22 relation types) for RGCN training, where each edge collection maps to a relation-type index. The codebase graph follows the same convention — collection names as relation indices:

```rust
pub const CODEBASE_EDGE_COLLECTIONS: &[&str] = &[
    "codebase_defines_edges",      // 0
    "codebase_calls_edges",        // 1
    "codebase_implements_edges",   // 2
    "codebase_imports_edges",      // 3
];
```

No translation layer. The `hades-prefetch` random walk code can use the same collection-name-to-relation-index pattern it already uses for the NL graph.

**Node features** for codebase vertices use a 5-value `kind` categorical (`file`, `module`, `type`, `callable`, `value`) — clean one-hot encoding, language-agnostic. The `lang_kind` string is available as an additional feature if finer granularity helps training.

---

## 10. Validation Rules

These invariants must hold after any ingest run completes successfully:

### Document Invariants

1. Every `codebase_chunks` document has a `file_key` that exists in `codebase_files`
2. Every `codebase_embeddings` document has a `chunk_key` that exists in `codebase_chunks`
3. Every `codebase_embeddings` document has a `file_key` that exists in `codebase_files`
4. `codebase_files.chunk_count` equals `COUNT(codebase_chunks WHERE file_key == _key)`
5. `codebase_files.embedding_count` equals `COUNT(codebase_embeddings WHERE file_key == _key)`
6. `codebase_files.symbol_count` equals `COUNT(codebase_symbols WHERE file_key == _key)` (syn-based count; `ra_symbol_count` may differ)
7. Every key in `codebase_chunks.symbols[]` exists in `codebase_symbols` — **post-ingest check only**, not a real-time constraint. Chunk documents are written during the syn pass; symbol documents may be overwritten by the rust-analyzer post-pass. This invariant is valid only after all ingest phases complete for a given file.

### Edge Invariants

8. Every `codebase_defines_edges._from` is in `codebase_files`; every `._to` is in `codebase_symbols`
9. Every `codebase_calls_edges._from` and `._to` are in `codebase_symbols`
10. Every `codebase_implements_edges._from` and `._to` are in `codebase_symbols`
11. Every `codebase_imports_edges._from` is in `codebase_files`; `._to` is in `codebase_symbols` or `codebase_files`

Note: invariants 8–11 are enforced by ArangoDB's named graph vertex constraints at insert time.

### Uniqueness Invariants

12. No two documents in any collection share the same `_key`
13. Edge keys are deterministic: the same `(from, kind, to)` triple always produces the same `_key`
14. Symbol keys are deterministic: the same `(file_key, qualified_name)` pair always produces the same `_key`

### Primitive Invariants

15. Every `codebase_files.kind` is `"file"`
16. Every `codebase_symbols.kind` is one of: `"module"`, `"type"`, `"callable"`, `"value"`
17. No symbol document exists with `lang_kind` of `"import"` or `"impl"` — these are not primitives

---

## 11. Open Questions

### Q1: Full-text index on chunk text?

ArangoDB supports `arangosearch` views for text search. Adding a view over `codebase_chunks.text` would enable keyword search without the embedding service.

**Trade-off:** Additional index maintenance overhead, but enables hybrid search (vector + keyword) directly in AQL instead of the current Rust-side hybrid reranker.

---

## 12. Collection Summary

```text
codebase_graph (Named Graph)
|
+-- codebase_files               (document, vertex)    — kind: file
+-- codebase_symbols             (document, vertex)    — kind: module|type|callable|value
+-- codebase_defines_edges       (edge: file -> symbol)
+-- codebase_calls_edges         (edge: symbol -> symbol)
+-- codebase_implements_edges    (edge: symbol -> symbol)
+-- codebase_imports_edges       (edge: file -> symbol|file)
|
+-- codebase_chunks              (document, orphan)    — text spans with symbol context
+-- codebase_embeddings          (document, orphan)    — embedding vectors
```

### Edge Collection Summary

| Collection | From | To | Count per file | Produced by |
|-----------|------|----|---------------|-------------|
| `codebase_defines_edges` | file | symbol | 1 per symbol | syn/AST extraction |
| `codebase_calls_edges` | callable | callable | 0..N per callable | rust-analyzer |
| `codebase_implements_edges` | callable | callable | 0..N per impl block | rust-analyzer |
| `codebase_imports_edges` | file | symbol or file | 0..N per import stmt | import resolution |

### Index Summary

| Collection | Indexed Fields | Purpose |
|-----------|---------------|---------|
| `codebase_chunks` | `file_key` | Cascade delete, chunk listing |
| `codebase_chunks` | `symbols[*]` | Find chunks containing a symbol |
| `codebase_embeddings` | `file_key` | Cascade delete, backfill check |
| `codebase_embeddings` | `chunk_key` | Vector search join |
| `codebase_symbols` | `file_key` | Cascade delete, symbol listing |
| `codebase_symbols` | `kind` | Primitive-filtered queries |

---

## Appendix A: Comparison with NL Graph Schema

| Aspect | NL Knowledge Graph | Codebase Graph |
|--------|-------------------|----------------|
| Named graphs | 6 (`nl_core`, `nl_equations`, ..., `nl_concept_map`) | 1 (`codebase_graph`) |
| Edge collections | 14 unique (16 defs, some share names) | 4 |
| Relation type discrimination | Collection name | Collection name |
| RGCN relation indices | `EDGE_COLLECTION_NAMES[0..21]` | `CODEBASE_EDGE_COLLECTIONS[0..3]` |
| Vertex collections | 80+ (paper-scoped x concept types) | 2 (files, symbols) |
| Vertex kind taxonomy | Per-collection (axioms, equations, definitions, ...) | 5 universal primitives |
| Orphan collections | 0 | 2 (chunks, embeddings) |
| Scope | Global (all papers, all workspaces) | Per-workspace |
| Scale | ~100K+ documents | ~10K documents per workspace |

## Appendix B: AQL Query Examples

### Traverse all symbols defined in a file

```aql
FOR v, e IN 1 OUTBOUND "codebase_files/src_lib_rs"
  codebase_defines_edges
  RETURN { symbol: v.qualified_name, kind: v.kind, lang_kind: v.lang_kind, line: v.start_line }
```

### Find all callers of a callable

```aql
FOR v, e IN 1 INBOUND "codebase_symbols/src_lib_rs__Config__new__a1b2c3d4"
  codebase_calls_edges
  RETURN { caller: v.qualified_name, file: v.file_path }
```

### Cross-file import chain (2 hops)

```aql
FOR v, e, p IN 1..2 OUTBOUND "codebase_files/src_main_rs"
  codebase_imports_edges
  RETURN { target: v.qualified_name OR v.path, depth: LENGTH(p.edges) }
```

### Find all callables exposed to Python

```aql
FOR sym IN codebase_symbols
  FILTER sym.is_pyo3 == true
  RETURN { name: sym.qualified_name, python_name: sym.python_name, file: sym.file_path }
```

### All types in the codebase (language-agnostic)

```aql
FOR sym IN codebase_symbols
  FILTER sym.kind == "type"
  RETURN { name: sym.qualified_name, lang_kind: sym.lang_kind, file: sym.file_path }
```

### Vector search with symbol context (no traversal needed)

```aql
FOR chunk_key IN @top_k_chunk_keys
  LET chunk = DOCUMENT(CONCAT("codebase_chunks/", chunk_key))
  LET file = DOCUMENT(CONCAT("codebase_files/", chunk.file_key))
  RETURN {
    text: chunk.text,
    file: file.path,
    language: file.language,
    symbols: chunk.symbols
  }
```

### Full graph walk (all edge types)

```aql
FOR v, e, p IN 1..3 ANY "codebase_files/src_lib_rs"
  GRAPH "codebase_graph"
  RETURN { vertex: v._key, kind: v.kind, depth: LENGTH(p.edges) }
```

---

## Appendix C: Decisions Log

| # | Question | Decision | Rationale |
|---|----------|----------|-----------|
| D1 | Symbol kind taxonomy | Universal 5-value primitives with `lang_kind` for original | Primitives first. Chunking, traversal, GNN features operate on primitives, not language artifacts. |
| D2 | Symbol context in chunks | Yes — `symbols: string[]` computed via interval intersection at ingest | Microseconds in Rust vs hundreds of ms as live AQL. Eliminates chunk→file→symbol table scan. |
| D3 | Single vs separate edge collections | Separate: 4 collections, one per relation | Aligns with NL graph convention. Collection-level skipping for random walks. Stronger from/to constraints via Gharial. No translation layer for RGCN. |
| D4 | Import symbols | Removed from `codebase_symbols` | Imports are not primitives. They produce edges, not vertices. |
| D5 | Impl block symbols | Removed from `codebase_symbols` | Impl blocks are scaffolding. Methods extracted individually as `callable` with `parent_symbol` pointing to the type. |
| D6 | `type` field on edge documents | Removed | Collection name IS the relation type. One source of truth. |
| D7 | `edge_key()` keeps `kind` param | Yes | Deterministic key stability. Same `(from, to)` pair in different collections must produce different keys. |
| D8 | Import `style` field | Added to `codebase_imports_edges` metadata | Low cost, enables dependency analysis granularity. |
| D9 | Embedding model hash | Yes — add `model_hash` to `codebase_embeddings` | Daemon holds active hash in memory via Persephone handshake. Incremental skip filters on both `symbol_hash` and `model_hash`. Enables selective re-embedding on model change without full re-ingest. |
