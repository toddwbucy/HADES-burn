"""Jina V4 embedding backend — the ML boundary for vector generation.

Wraps jinaai/jina-embeddings-v4 behind a minimal interface.
The model is Qwen2.5-VL-3B-Instruct with 3 task-specific LoRA adapters
(retrieval, text-matching, code). trust_remote_code is required because
Jina forks the entire Qwen forward pass to route task_label through
every layer for adapter selection.

Everything else (chunking, batching, DB writes) is handled by the Rust
orchestrator.
"""

from __future__ import annotations

import gc
import logging
import time
from typing import Any

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)

# Jina V4 constants
EMBEDDING_DIM = 2048
MAX_TOKENS = 32768

# Task label mapping: proto task string → Jina LoRA adapter name
_TASK_TO_ADAPTER = {
    "retrieval.passage": "retrieval",
    "retrieval.query": "retrieval",
    "retrieval": "retrieval",
    "text-matching": "text-matching",
    "code": "code",
}

# Supported task labels exposed via Info RPC
SUPPORTED_TASKS = ["retrieval.passage", "retrieval.query", "text-matching", "code"]


class JinaV4Embedder:
    """Embed text using Jina V4 (Qwen2.5-VL-3B + LoRA adapters).

    Lazy-loads the model on first use. Call unload() to release GPU memory.
    """

    def __init__(
        self,
        *,
        model_name: str = "jinaai/jina-embeddings-v4",
        device: str = "cuda:2",
        use_fp16: bool = True,
        batch_size: int = 128,
    ) -> None:
        self._model_name = model_name
        self._batch_size = batch_size

        # Validate device
        if device.startswith("cuda") and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, falling back to CPU")
            self._device = "cpu"
        else:
            self._device = device

        self._dtype = (
            torch.float16
            if (use_fp16 and self._device.startswith("cuda"))
            else torch.float32
        )

        self._model = None
        self._tokenizer = None

    def _load(self) -> None:
        """Load model and tokenizer into GPU memory."""
        logger.info(
            "Loading %s on %s (dtype=%s)", self._model_name, self._device, self._dtype
        )
        start = time.time()

        self._tokenizer = AutoTokenizer.from_pretrained(
            self._model_name, trust_remote_code=True
        )

        self._model = AutoModel.from_pretrained(
            self._model_name, trust_remote_code=True, torch_dtype=self._dtype
        )

        if self._device != "cpu":
            self._model = self._model.to(self._device)

        # Set model to inference mode
        self._model.requires_grad_(False)
        logger.info(
            "Model loaded in %.2fs (dtype=%s)",
            time.time() - start,
            next(self._model.parameters()).dtype,
        )

    @property
    def model(self):
        """Lazy-load model on first access."""
        if self._model is None:
            self._load()
        return self._model

    @property
    def tokenizer(self):
        """Lazy-load tokenizer on first access."""
        if self._tokenizer is None:
            self._load()
        return self._tokenizer

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def embedding_dimension(self) -> int:
        return EMBEDDING_DIM

    @property
    def max_sequence_length(self) -> int:
        return MAX_TOKENS

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def embed_texts(
        self,
        texts: list[str],
        task: str = "retrieval.passage",
        batch_size: int | None = None,
    ) -> np.ndarray:
        """Embed texts using Jina V4.

        Args:
            texts: List of texts to embed.
            task: Task type controlling LoRA adapter selection.
            batch_size: Override default batch size.

        Returns:
            Numpy array of shape (N, 2048) with float32 embeddings.
        """
        all_embeddings: list[np.ndarray] = []
        batch_size = batch_size or self._batch_size

        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                embeddings = self._embed_batch(batch, task)
                all_embeddings.append(embeddings)

        if all_embeddings:
            return np.vstack(all_embeddings).astype(np.float32, copy=False)
        return np.empty((0, EMBEDDING_DIM), dtype=np.float32)

    def _embed_batch(self, batch: list[str], task: str) -> np.ndarray:
        """Embed a single batch, choosing the best available API."""
        model = self.model  # triggers lazy load

        # Prefer Jina's high-level encode_text() when available
        if hasattr(model, "encode_text"):
            jina_task = _TASK_TO_ADAPTER.get(task, "retrieval")
            prompt_name = "query" if task == "retrieval.query" else "passage"
            embeddings = model.encode_text(
                batch, task=jina_task, prompt_name=prompt_name
            )
        else:
            # Fallback: raw forward pass with task_label for LoRA selection
            task_label = _TASK_TO_ADAPTER.get(task, "retrieval")
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MAX_TOKENS,
            ).to(self._device)

            outputs = model(**inputs, task_label=task_label)

            if hasattr(outputs, "single_vec_emb") and outputs.single_vec_emb is not None:
                embeddings = outputs.single_vec_emb
            else:
                available = [a for a in dir(outputs) if not a.startswith("_")]
                raise AttributeError(
                    f"Expected 'single_vec_emb' in output, got: {available}"
                )

        return self._to_numpy(embeddings)

    @staticmethod
    def _to_numpy(embeddings: Any) -> np.ndarray:
        """Convert embeddings (tensor, list, or ndarray) to numpy float32."""
        if torch.is_tensor(embeddings):
            if embeddings.is_cuda:
                embeddings = embeddings.cpu()
            return embeddings.numpy().astype(np.float32, copy=False)

        if hasattr(embeddings, "detach"):
            embeddings = embeddings.detach()
            if hasattr(embeddings, "is_cuda") and embeddings.is_cuda:
                embeddings = embeddings.cpu()
            return embeddings.numpy().astype(np.float32, copy=False)

        if isinstance(embeddings, list):
            processed = []
            for e in embeddings:
                if torch.is_tensor(e):
                    if e.is_cuda:
                        e = e.cpu()
                    processed.append(e.numpy())
                else:
                    processed.append(np.array(e))
            return np.vstack(processed).astype(np.float32, copy=False)

        if isinstance(embeddings, np.ndarray):
            return embeddings.astype(np.float32, copy=False)

        return np.array(embeddings, dtype=np.float32)

    def unload(self) -> None:
        """Release GPU memory held by the model."""
        if self._model is not None:
            logger.info("Unloading Jina V4 model...")
            del self._model
            self._model = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        gc.collect()
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("CUDA cache cleared")
        except Exception:
            pass
        logger.info("Jina V4 model unloaded")
