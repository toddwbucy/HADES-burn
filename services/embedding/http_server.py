"""Persephone Embedding Service — OpenAI-compatible HTTP server.

Wraps `JinaV4Embedder` behind a FastAPI app exposing the OpenAI
`/v1/embeddings` and `/v1/models` endpoints. Replaces the prior gRPC
server (`server.py`) — same inference layer, different transport.

Listens on TCP `127.0.0.1:8000` by default. HADES's embedding client
already speaks OpenAI shape (post-refactor), so configuration is just a
matter of pointing it at this service's URL.

The `task` field on the request body is a HADES/Jina vendor extension:
it routes to the right LoRA adapter (`retrieval.query`,
`retrieval.passage`, `text-matching`, `code`). OpenAI clients that
don't send it default to `retrieval.passage`.

Usage:
    python -m embedding.http_server
"""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import Any, Optional, Union

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .config import EmbeddingConfig
from .jina_v4 import EMBEDDING_DIM, MAX_TOKENS, SUPPORTED_TASKS, JinaV4Embedder

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Request / response models (OpenAI-compatible shapes)
# ---------------------------------------------------------------------------


class EmbedRequest(BaseModel):
    """OpenAI-compatible request body for `POST /v1/embeddings`.

    `task` and `batch_size` are non-standard vendor extensions used by
    Jina V4 (LoRA adapter routing) and HADES (server-side batching hint).
    Engines that don't recognize them ignore them.
    """

    model: str
    input: Union[str, list[str]]
    encoding_format: str = "float"
    # Vendor extensions
    task: Optional[str] = Field(
        default="retrieval.passage",
        description="Jina V4 LoRA adapter selector",
    )
    batch_size: Optional[int] = Field(
        default=None,
        description="Server-side batch size override",
    )


class EmbedItem(BaseModel):
    object: str = "embedding"
    embedding: list[float]
    index: int


class Usage(BaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0


class EmbedResponse(BaseModel):
    object: str = "list"
    data: list[EmbedItem]
    model: str
    usage: Usage


class ModelInfo(BaseModel):
    """Single entry in `/v1/models` response.

    The OpenAI-standard fields are `id`, `object`, `created`, `owned_by`.
    `dimension`, `max_seq_length`, and `supported_tasks` are vendor
    extensions HADES surfaces so clients can discover model capabilities
    in one round-trip.
    """

    id: str
    object: str = "model"
    created: int = 0
    owned_by: str = "hades"
    # Vendor extensions
    dimension: int = EMBEDDING_DIM
    max_seq_length: int = MAX_TOKENS
    supported_tasks: list[str] = Field(default_factory=lambda: list(SUPPORTED_TASKS))


class ModelsResponse(BaseModel):
    object: str = "list"
    data: list[ModelInfo]


# ---------------------------------------------------------------------------
# Shared application state
# ---------------------------------------------------------------------------


class AppState:
    """State held across requests: the embedder, request bookkeeping,
    and the idle-monitor task."""

    def __init__(self, config: EmbeddingConfig) -> None:
        self.config = config
        # Eager construction; the underlying model loads lazily on first embed.
        self.embedder = JinaV4Embedder(
            model_name=config.model_name,
            device=config.device,
            use_fp16=config.use_fp16,
            batch_size=config.batch_size,
        )
        self.last_request_time = time.time()
        self.active_requests = 0
        self._idle_monitor_task: Optional[asyncio.Task[None]] = None

    async def start_idle_monitor(self) -> None:
        if self.config.idle_timeout_seconds > 0:
            self._idle_monitor_task = asyncio.create_task(
                self._idle_monitor(), name="embedder-idle-monitor"
            )

    async def stop_idle_monitor(self) -> None:
        if self._idle_monitor_task is not None:
            self._idle_monitor_task.cancel()
            try:
                await self._idle_monitor_task
            except asyncio.CancelledError:
                pass

    async def _idle_monitor(self) -> None:
        """Unload the model from VRAM after the configured idle window.

        Mirrors the behavior of the prior gRPC server: poll on a bounded
        cadence, only unload when no requests are in flight, only when the
        model is actually loaded, and only when the last request is older
        than the threshold.
        """
        poll_interval = min(60, max(1, int(self.config.idle_timeout_seconds)))
        while True:
            await asyncio.sleep(poll_interval)
            if (
                self.active_requests == 0
                and self.embedder.is_loaded
                and (time.time() - self.last_request_time)
                > self.config.idle_timeout_seconds
            ):
                logger.info(
                    "Embedder idle for %.0fs — unloading model to free GPU memory",
                    self.config.idle_timeout_seconds,
                )
                self.embedder.unload()


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: build state, run, tear down cleanly."""
    config = EmbeddingConfig.from_env()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    state = AppState(config)
    app.state.app_state = state
    await state.start_idle_monitor()
    logger.info(
        "Embedding service ready (model=%s, device=%s, idle_timeout=%.0fs, listen=%s:%d)",
        config.model_name,
        config.device,
        config.idle_timeout_seconds,
        config.host,
        config.port,
    )
    try:
        yield
    finally:
        logger.info("Shutting down embedding service")
        await state.stop_idle_monitor()
        if state.embedder.is_loaded:
            state.embedder.unload()
        logger.info("Embedding service stopped")


app = FastAPI(
    title="HADES Persephone Embedding Service",
    description="OpenAI-compatible /v1/embeddings server backed by Jina V4",
    version="0.3.0",
    lifespan=lifespan,
)


def _state(app: FastAPI) -> AppState:
    return app.state.app_state  # type: ignore[no-any-return]


@app.get("/v1/models", response_model=ModelsResponse)
async def list_models() -> ModelsResponse:
    """OpenAI `/v1/models` — single entry for the configured Jina V4 model."""
    state = _state(app)
    return ModelsResponse(data=[ModelInfo(id=state.config.model_name)])


@app.post("/v1/embeddings", response_model=EmbedResponse)
async def create_embeddings(req: EmbedRequest) -> EmbedResponse:
    """OpenAI `/v1/embeddings` — embeds one or more inputs as 2048-dim vectors.

    The `task` vendor-extension field selects the Jina V4 LoRA adapter.
    Defaults to `retrieval.passage` if absent. Unknown tasks are rejected
    (400) rather than silently routed to a wrong adapter.
    """
    state = _state(app)

    # Normalize input (single string or list of strings)
    texts: list[str] = [req.input] if isinstance(req.input, str) else list(req.input)

    if not texts:
        raise HTTPException(status_code=400, detail="`input` must be non-empty")

    task = req.task or "retrieval.passage"
    if task not in SUPPORTED_TASKS:
        raise HTTPException(
            status_code=400,
            detail=f"unknown task {task!r}; supported: {', '.join(SUPPORTED_TASKS)}",
        )

    if req.batch_size is not None and req.batch_size < 0:
        raise HTTPException(
            status_code=400,
            detail=f"`batch_size` must be non-negative, got {req.batch_size}",
        )
    batch_override = (
        req.batch_size if (req.batch_size is not None and req.batch_size > 0) else None
    )

    state.active_requests += 1
    started = time.time()
    try:
        loop = asyncio.get_running_loop()
        # Inference is sync (PyTorch); run in the default executor pool to
        # avoid blocking the event loop. Don't share a future across requests
        # — each call creates its own.
        vectors = await loop.run_in_executor(
            None,
            lambda: state.embedder.embed_texts(
                texts, task=task, batch_size=batch_override
            ),
        )
    except ValueError as e:
        # `JinaV4Embedder.embed_texts` raises ValueError for invalid
        # batch_size; surface that as a 400 not a 500.
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Embedding failed")
        raise HTTPException(status_code=500, detail=f"embedding failed: {e}")
    finally:
        state.active_requests -= 1
        state.last_request_time = time.time()

    duration_ms = int((time.time() - started) * 1000)
    logger.info(
        "Embed: %d texts, task=%s, %dms", len(texts), task, duration_ms
    )

    items = [
        EmbedItem(embedding=row.tolist(), index=i)
        for i, row in enumerate(vectors)
    ]
    return EmbedResponse(
        data=items,
        model=state.embedder.model_name,
        usage=Usage(),  # OpenAI sends real token counts; we don't track them yet.
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run uvicorn against this app, configured from env vars.

    Single-process by design: each worker would load its own copy of the
    Jina V4 model into VRAM, exhausting the GPU. If you need higher
    throughput, use larger batches via `batch_size` in the request body.
    """
    config = EmbeddingConfig.from_env()
    uvicorn.run(
        "embedding.http_server:app",
        host=config.host,
        port=config.port,
        log_level="info",
        reload=False,
        workers=1,
    )


if __name__ == "__main__":
    main()
