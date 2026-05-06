"""Embedding service configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class EmbeddingConfig:
    """Configuration for the embedding service.

    The HTTP server binds to (host, port). Unix-domain-socket binding was
    supported by an earlier gRPC implementation; that path is intentionally
    omitted now — HADES's latency margin is generous enough that TCP
    localhost is fine, and the simpler config surface is worth more than
    UDS would buy here.
    """

    host: str = "127.0.0.1"
    port: int = 8000
    device: str = "cuda:2"
    model_name: str = "jinaai/jina-embeddings-v4"
    use_fp16: bool = True
    batch_size: int = 128
    idle_timeout_seconds: float = 900.0  # 15 min — unload model after idle

    @classmethod
    def from_env(cls) -> EmbeddingConfig:
        raw_batch = os.environ.get("HADES_EMBEDDER_BATCH_SIZE", "")
        raw_idle = os.environ.get("HADES_EMBEDDER_IDLE_TIMEOUT", "")
        raw_port = os.environ.get("HADES_EMBEDDER_PORT", "")

        try:
            batch_size = int(raw_batch) if raw_batch else cls.batch_size
        except ValueError:
            raise ValueError(
                f"HADES_EMBEDDER_BATCH_SIZE must be an integer, got {raw_batch!r}"
            ) from None

        try:
            idle_timeout = float(raw_idle) if raw_idle else cls.idle_timeout_seconds
        except ValueError:
            raise ValueError(
                f"HADES_EMBEDDER_IDLE_TIMEOUT must be a number, got {raw_idle!r}"
            ) from None

        try:
            port = int(raw_port) if raw_port else cls.port
        except ValueError:
            raise ValueError(
                f"HADES_EMBEDDER_PORT must be an integer, got {raw_port!r}"
            ) from None

        return cls(
            host=os.environ.get("HADES_EMBEDDER_HOST", cls.host),
            port=port,
            device=os.environ.get("HADES_EMBEDDER_DEVICE", cls.device),
            model_name=os.environ.get(
                "HADES_EMBEDDER_MODEL", cls.model_name
            ),
            use_fp16=os.environ.get(
                "HADES_EMBEDDER_FP16", str(cls.use_fp16)
            ).lower()
            in ("1", "true", "yes"),
            batch_size=batch_size,
            idle_timeout_seconds=idle_timeout,
        )
