# Persephone Embedding API Specification

**Version:** 1.0
**Status:** Proposed (PR #71)
**Path prefix:** `/v1`

## Overview

The Persephone Embedding API is HADES's contract for embedding service backends. It is **inspired by but not compatible with** OpenAI's `/v1/embeddings` API: requests share the same shape, but responses always return **late-chunked output** with per-chunk metadata. There is no single-vector-per-input mode.

HADES is **engine-agnostic** at this contract level — any backend that implements PE-API is a valid backend (the FastAPI service in this repo, a future Rust-native loader, a `hades-weaver-bridge` translator, an external service). HADES is **model-bound** at the data layer: implementations must serve a model with Jina V4's capability profile (2048-dim, 32k context, multimodal, task-conditional via LoRA, late-chunking-capable). Wrong model class → silently incompatible vector geometry.

### Why diverge from OpenAI

OpenAI's `/v1/embeddings` returns one vector per input. HADES depends on **late chunking** — full-document encoding followed by chunk-aware pooling — for retrieval quality on documents above ~500 tokens. That returns *N* chunks per input, with each chunk's embedding informed by the full surrounding context. The shape mismatch is fundamental, not papered-over by a vendor-extension flag.

Trade-off accepted: PE-API clients are not portable to vanilla OpenAI servers, and OpenAI clients aren't portable to PE-API. The two ecosystems share request-construction conventions but diverge on response semantics.

## Endpoints

### `GET /v1/models`

Discover the model the backend has loaded. OpenAI-compatible response shape.

**Request:** none.

**Response:**

```json
{
  "object": "list",
  "data": [
    {
      "id": "jinaai/jina-embeddings-v4",
      "object": "model",
      "created": 0,
      "owned_by": "hades",

      "dimension": 2048,
      "max_seq_length": 32768,
      "supported_tasks": [
        "retrieval.passage",
        "retrieval.query",
        "text-matching",
        "code"
      ]
    }
  ]
}
```

`dimension`, `max_seq_length`, and `supported_tasks` are **PE-API extensions** to OpenAI's model object.

- **`supported_tasks` is REQUIRED.** Conforming backends MUST include it, accurately reflecting the tasks they can serve (see Implementation requirements). Clients use this field to validate `task` values before sending requests, avoiding unnecessary round-trips that would only fail with `PE_INVALID_TASK`.
- **`dimension` and `max_seq_length` are OPTIONAL.** Backends MAY omit them; clients that need either value can determine it via an embedding round-trip (`/v1/embeddings` with a probe input).

### `POST /v1/embeddings`

Primary embedding endpoint. Always late-chunked. Shape diverges from OpenAI.

**Request:**

```jsonc
{
  "model": "jinaai/jina-embeddings-v4",       // OpenAI standard
  "input": "long document text...",           // OpenAI standard: string OR array of strings
  "encoding_format": "float",                 // OpenAI standard; only "float" supported in v1.0

  // PE-API extensions:
  "task": "retrieval.passage",                // LoRA adapter selector (see "Tasks" below)
  "images": ["base64-encoded image", ...],    // multimodal inputs paired with `input` strings
  "chunk_size_tokens": 500,                   // late-chunking target chunk size (default: backend-configured)
  "chunk_overlap_tokens": 200                 // late-chunking inter-chunk overlap
}
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `model` | string | yes | — | Model identifier; must match a model from `GET /v1/models` |
| `input` | string \| string[] | yes | — | Text(s) to embed. Each input is independently late-chunked |
| `encoding_format` | string | no | `"float"` | Reserved for future encoding variants |
| `task` | string | no | `"retrieval.passage"` | Jina V4 LoRA adapter; see Tasks |
| `images` | string[] | no | `[]` | Per-input base64-encoded images for multimodal embedding. When provided, `len(images)` must equal the number of inputs — `1` if `input` is a string, otherwise `len(input)` |
| `chunk_size_tokens` | int | no | backend-configured | Late-chunking chunk size in tokens |
| `chunk_overlap_tokens` | int | no | backend-configured | Late-chunking overlap in tokens |

**Response:**

```jsonc
{
  "object": "list",
  "data": [
    {
      "object": "embedding",
      "index": 0,                             // index into request `input` array
      "embedding": [0.0123, -0.0456, ...],    // length = model dimension (2048 for Jina V4)

      // PE-API extension: chunk metadata
      "chunk": {
        "text": "the actual chunk text...",
        "start_char": 0,                      // offsets in original `input[index]`
        "end_char": 487,
        "start_token": 0,
        "end_token": 195,
        "chunk_index": 0,                     // chunk position within input[index]
        "total_chunks": 5,                    // total chunks produced for input[index]
        "context_window_used": 487            // tokens of context this embedding was conditioned on
      }
    },
    {
      "object": "embedding",
      "index": 0,                             // same input, next chunk
      "embedding": [...],
      "chunk": {
        "chunk_index": 1,
        "total_chunks": 5,
        // ...
      }
    }
    // ...
  ],
  "model": "jinaai/jina-embeddings-v4",
  "usage": {
    "prompt_tokens": 2400,
    "total_tokens": 2400
  }
}
```

**Critical shape difference from OpenAI:** for a single-string `input`, `data` may contain *multiple entries* — one per chunk produced by late chunking. Each entry's `index` field references its position in the request's `input` array; multiple chunks share the same `index` when they came from the same input string. Use `chunk.chunk_index` and `chunk.total_chunks` to identify chunk position within the input.

**Short-input behavior:** when an input fits in a single chunk, the response contains exactly one `data` entry for that input with `chunk.chunk_index = 0` and `chunk.total_chunks = 1`. This is the default behavior for queries and short text — there is no separate "vanilla embedding" path.

## Tasks (Jina V4 LoRA adapter routing)

The `task` field selects which LoRA adapter Jina V4 uses to produce embeddings. Different adapters produce different geometries; **mismatched task between query and corpus drastically degrades retrieval quality.**

| Task | Adapter | Use case |
|------|---------|----------|
| `retrieval.passage` | `retrieval` | Embedding documents for retrieval (default) |
| `retrieval.query` | `retrieval` | Embedding search queries (paired with `retrieval.passage` corpus) |
| `text-matching` | `text-matching` | Symmetric text similarity (paired-document scoring) |
| `code` | `code` | Code embedding |

Backends that don't honor `task` (e.g., a backend serving Jina V4 with a single pre-baked adapter) **must reject requests with unsupported `task` values** with HTTP 400, rather than silently routing to the wrong adapter and producing geometrically incomparable vectors.

## Multimodal (text + images)

When `images` is provided, it must be the same length as `input`. Each `input[i]` is paired with `images[i]` as a multimodal input. `images[i]` is a base64-encoded image (PNG, JPEG; format detected from bytes).

A backend that does not support multimodal **must** reject requests with non-empty `images` with HTTP 400, rather than silently ignoring images and producing text-only embeddings (which would be silently lower-quality and cohort-mismatched against multimodal-corpus data).

## Error handling

Errors follow OpenAI's error envelope:

```json
{
  "error": {
    "message": "human-readable description",
    "type": "invalid_request_error" | "server_error",
    "param": "task" | "input" | null,
    "code": "PE_INVALID_TASK" | "PE_DIMENSION_MISMATCH" | ...
  }
}
```

By convention, **4xx** HTTP responses (caller's fault) use `type: "invalid_request_error"`, and **5xx** responses (server's fault) use `type: "server_error"`. Unclassified or library-specific errors default to `"server_error"`. The `code` field is PE-API-specific and provides a more granular machine-readable identifier than `type`.

PE-API-specific error codes:

| Code | HTTP | Meaning |
|------|------|---------|
| `PE_INVALID_TASK` | 400 | `task` value not in backend's `supported_tasks` |
| `PE_UNSUPPORTED_ENCODING_FORMAT` | 400 | `encoding_format` value other than `"float"` (only `"float"` is supported in v1.0) |
| `PE_MULTIMODAL_UNSUPPORTED` | 400 | `images` provided but backend serves text-only model |
| `PE_INPUT_IMAGES_LENGTH_MISMATCH` | 400 | `len(images) != input_count`, where `input_count = 1` if `input` is a string, else `len(input)` |
| `PE_INPUT_TOO_LARGE` | 400 | An input exceeds backend's max-context fallback handling |
| `PE_MODEL_NOT_LOADED` | 503 | Model is unloaded (e.g., idle-timeout); retry after warm-up |
| `PE_BACKEND_OOM` | 503 | GPU OOM on this batch; retry with smaller batch or wait |

## Cohort identity (sketched, deferred to v1.1)

A `cohort` field on the response is reserved for v1.1+. Its purpose is to enable forensic and cross-corpus queries to verify that two vector sets are geometrically comparable before running similarity operations.

```jsonc
// v1.1 response (sketched, not implemented in v1.0):
{
  // ... v1.0 fields ...
  "cohort": {
    "model": "jinaai/jina-embeddings-v4",
    "model_revision": "<HF model revision SHA>",
    "lora_adapter": "retrieval",
    "tokenizer_hash": "<hash>",
    "fa2_kernel_version": "<candle-flash-attn build identifier>",
    "precision": "bf16"
  }
}
```

Concrete shape lands once HADES's forensic / multi-cohort query work begins (see `project_weaver_embedder_cohort_pin.md`). v1.0 backends should record the equivalent metadata internally; surfacing it on the wire is v1.1.

## Implementation requirements

A conforming PE-API v1.0 backend MUST:

1. Implement `GET /v1/models` returning at least one model with the Jina V4 capability profile (2048-dim, 32k context, late-chunking-capable). The model's `supported_tasks` field is REQUIRED and MUST accurately list the tasks the backend can serve — neither over- nor under-claiming.
2. Implement `POST /v1/embeddings` returning chunk-array shape — even for single-chunk inputs.
3. Honor `task` for adapter selection when the value is listed in the model's `supported_tasks`. Reject `task` values not in `supported_tasks` with `PE_INVALID_TASK`. Silently routing to a different adapter is forbidden — the cost of "wrong adapter" silently degraded retrieval quality is much higher than a hard error.
4. Reject multimodal requests it cannot serve (i.e., `images` provided to a text-only backend) with `PE_MULTIMODAL_UNSUPPORTED` rather than silently producing text-only embeddings.
5. Reject `encoding_format` values other than `"float"` with `PE_UNSUPPORTED_ENCODING_FORMAT`. v1.0 supports only `"float"`; the field is reserved for future variants.
6. Return HTTP 503 with `PE_MODEL_NOT_LOADED` when the model is unloaded; do not block clients on model warm-up.

A conforming PE-API v1.0 backend SHOULD:

- Support all four standard tasks (`retrieval.passage`, `retrieval.query`, `text-matching`, `code`) where the loaded model is capable. Backends limited to a single pre-baked adapter are valid (they list a single task in `supported_tasks` and reject the others) but reduce HADES's retrieval quality and are NOT the recommended deployment shape.

A conforming backend MAY:

- Surface `device`, `dimension`, and `max_seq_length` in `/v1/models` response (these fields are optional; `supported_tasks` is required per item 1 above).
- Pre-allocate / batch internally on top of the per-request `chunk_size_tokens`/`chunk_overlap_tokens` overrides.
- Idle-unload the model after configurable timeout, returning `PE_MODEL_NOT_LOADED` until reload.

## Compatibility with OpenAI ecosystem

The OpenAI Python and TypeScript client libraries can construct PE-API requests because the request shape is a superset of OpenAI's. They **cannot** correctly interpret PE-API responses because:

- Response `data` may have more entries than `input` had elements (one entry per chunk, not per input).
- Each `data` entry has a `chunk` field OpenAI clients don't know how to handle.

Clients consuming PE-API are expected to use a PE-API-aware client (e.g., the Rust client at `crates/hades-core/src/persephone/embedding.rs`). Pointing an OpenAI client at a PE-API endpoint is **not supported** and will surface as silent index/chunk confusion.

## Reference implementation

The FastAPI service at `services/embedding/http_server.py` is the v1.0 reference implementation. The Rust client at `crates/hades-core/src/persephone/embedding.rs` is the v1.0 reference consumer. Both are normative for behavior questions not resolved by this spec.

## Future direction

- **v1.1:** add `cohort` field to response (concrete shape).
- **v1.2:** streaming responses for very-long-document embedding (reduces TTFB on >100k-token inputs).
- **v2:** consider whether the chunk-array shape should split into a separate path (`/v1/late_chunk_embeddings`) and `/v1/embeddings` should restore strict OpenAI compatibility for use cases that don't need late chunking. Decision deferred until we have data on how often (if ever) HADES's vanilla-embedding path is actually wanted.
