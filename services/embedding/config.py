"""Embedding service configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class EmbeddingConfig:
    """Configuration for the embedding service."""

    socket_path: str = "/run/hades/embedder.sock"
    device: str = "cuda:2"
    model_name: str = "jinaai/jina-embeddings-v4"
    use_fp16: bool = True
    batch_size: int = 128
    idle_timeout_seconds: float = 900.0  # 15 min — unload model after idle

    @classmethod
    def from_env(cls) -> EmbeddingConfig:
        return cls(
            socket_path=os.environ.get(
                "HADES_EMBEDDER_SOCKET", cls.socket_path
            ),
            device=os.environ.get("HADES_EMBEDDER_DEVICE", cls.device),
            model_name=os.environ.get(
                "HADES_EMBEDDER_MODEL", cls.model_name
            ),
            use_fp16=os.environ.get("HADES_EMBEDDER_FP16", "true").lower()
            in ("1", "true", "yes"),
            batch_size=int(
                os.environ.get("HADES_EMBEDDER_BATCH_SIZE", cls.batch_size)
            ),
            idle_timeout_seconds=float(
                os.environ.get(
                    "HADES_EMBEDDER_IDLE_TIMEOUT", cls.idle_timeout_seconds
                )
            ),
        )
