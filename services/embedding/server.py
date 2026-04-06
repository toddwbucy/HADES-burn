"""Persephone Embedding Service — gRPC server.

Implements the EmbeddingService proto contract. Wraps JinaV4Embedder
behind gRPC, managing GPU memory lifecycle with idle unloading.

Usage:
    python -m embedding.server
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import stat
import sys
import time
from pathlib import Path

import grpc
from grpc import aio as grpc_aio

# Ensure generated stubs are importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "generated"))

from persephone.embedding import embedding_pb2, embedding_pb2_grpc  # noqa: E402

from .config import EmbeddingConfig  # noqa: E402
from .jina_v4 import EMBEDDING_DIM, MAX_TOKENS, SUPPORTED_TASKS, JinaV4Embedder  # noqa: E402

logger = logging.getLogger(__name__)


class EmbeddingServicer(embedding_pb2_grpc.EmbeddingServiceServicer):
    """gRPC servicer implementing the EmbeddingService proto."""

    def __init__(self, config: EmbeddingConfig) -> None:
        self._config = config
        # Eagerly create the embedder wrapper (model loads lazily on first request)
        self._embedder = JinaV4Embedder(
            model_name=config.model_name,
            device=config.device,
            use_fp16=config.use_fp16,
            batch_size=config.batch_size,
        )
        self._last_request_time = time.time()
        self._active_requests = 0
        self._executor_futures: set[asyncio.Future] = set()

    async def Embed(self, request, context):
        """Embed a batch of texts into vectors."""
        if not request.texts:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("No texts provided")
            return embedding_pb2.EmbedResponse()

        task = request.task or "retrieval.passage"
        if task not in SUPPORTED_TASKS:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(
                f"Unknown task {task!r}; supported: {', '.join(SUPPORTED_TASKS)}"
            )
            return embedding_pb2.EmbedResponse()

        logger.info(
            "Embed request: %d texts, task=%s", len(request.texts), task
        )

        if request.batch_size < 0:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(f"batch_size must be non-negative, got {request.batch_size}")
            return embedding_pb2.EmbedResponse()
        batch_size = request.batch_size if request.batch_size > 0 else None
        self._active_requests += 1

        try:
            loop = asyncio.get_running_loop()
            start = time.time()
            fut = loop.run_in_executor(
                None,
                lambda: self._embedder.embed_texts(
                    list(request.texts), task=task, batch_size=batch_size
                ),
            )
            self._executor_futures.add(fut)
            fut.add_done_callback(self._executor_futures.discard)
            vectors = await fut
            duration_ms = int((time.time() - start) * 1000)
        except Exception as e:
            logger.error("Embedding failed: %s", e)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Embedding failed: {e}")
            return embedding_pb2.EmbedResponse()
        finally:
            self._active_requests -= 1
            self._last_request_time = time.time()

        embeddings = [
            embedding_pb2.Embedding(values=row.tolist()) for row in vectors
        ]

        return embedding_pb2.EmbedResponse(
            embeddings=embeddings,
            model=self._embedder.model_name,
            dimension=EMBEDDING_DIM,
            duration_ms=duration_ms,
        )

    async def Info(self, request, context):
        """Report embedding provider capabilities."""
        return embedding_pb2.ProviderInfo(
            model_name=self._config.model_name,
            dimension=EMBEDDING_DIM,
            max_seq_length=MAX_TOKENS,
            supported_tasks=SUPPORTED_TASKS,
            device=self._embedder.device,
            model_loaded=self._embedder.is_loaded,
        )

    @property
    def last_request_time(self) -> float:
        return self._last_request_time

    def unload_model(self) -> None:
        """Unload the embedding model to free GPU memory."""
        if self._executor_futures:
            logger.info("Skipping unload — %d executor futures in flight", len(self._executor_futures))
            return
        if self._embedder.is_loaded:
            self._embedder.unload()
            logger.info("Embedding model unloaded")


async def idle_monitor(
    servicer: EmbeddingServicer, config: EmbeddingConfig
) -> None:
    """Background task to unload model after idle timeout."""
    poll_interval = (
        min(60, max(1, int(config.idle_timeout_seconds)))
        if config.idle_timeout_seconds > 0
        else 60
    )
    while True:
        await asyncio.sleep(poll_interval)
        if (
            config.idle_timeout_seconds > 0
            and servicer._active_requests == 0
            and servicer._embedder.is_loaded
            and (time.time() - servicer.last_request_time)
            > config.idle_timeout_seconds
        ):
            logger.info(
                "Embedder idle for %.0fs — unloading model to free GPU memory",
                config.idle_timeout_seconds,
            )
            servicer.unload_model()


async def serve() -> None:
    """Start the embedding gRPC server."""
    config = EmbeddingConfig.from_env()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    servicer = EmbeddingServicer(config)

    server = grpc_aio.server()
    embedding_pb2_grpc.add_EmbeddingServiceServicer_to_server(servicer, server)

    socket_path = config.socket_path
    if not socket_path:
        raise ValueError("HADES_EMBEDDER_SOCKET must be a non-empty path")

    socket_dir = Path(socket_path).parent
    socket_dir.mkdir(parents=True, exist_ok=True)

    # Clean stale socket (only if it's actually a socket)
    sock = Path(socket_path)
    if sock.exists():
        if stat.S_ISSOCK(sock.stat().st_mode):
            sock.unlink()
        else:
            raise RuntimeError(
                f"Path {socket_path} exists but is not a socket; refusing to remove"
            )

    server.add_insecure_port(f"unix:{socket_path}")

    logger.info("Starting embedding service on %s", socket_path)
    await server.start()

    # Start idle monitor
    monitor = asyncio.create_task(idle_monitor(servicer, config))

    # Handle shutdown signals
    stop_event = asyncio.Event()

    def _signal_handler():
        logger.info("Shutdown signal received")
        stop_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, _signal_handler)

    logger.info("Embedding service ready")
    await stop_event.wait()

    logger.info("Shutting down embedding service...")
    monitor.cancel()
    try:
        await monitor
    except asyncio.CancelledError:
        pass
    await server.stop(grace=5)
    servicer.unload_model()

    # Clean up socket
    if sock.exists() and stat.S_ISSOCK(sock.stat().st_mode):
        sock.unlink()

    logger.info("Embedding service stopped")


if __name__ == "__main__":
    asyncio.run(serve())
