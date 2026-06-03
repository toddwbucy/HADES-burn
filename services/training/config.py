"""Training service configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class TrainingConfig:
    """Configuration for the RGCN training service.

    The GPU is configurable: it defaults to the embedder's GPU (``cuda:2``)
    but can be overridden via ``HADES_TRAINER_DEVICE`` (or ``CUDA_VISIBLE_DEVICES``
    at the process level) to accommodate different hardware layouts. The device
    sent in an ``InitModel`` request takes precedence over this default when
    non-empty, so the client (driven by ``hades.yaml``'s ``gpu.device``) has the
    final say per training run.
    """

    socket_path: str = "/run/hades/training.sock"
    device: str = "cuda:2"

    @classmethod
    def from_env(cls) -> TrainingConfig:
        # `or` so a blank env entry (e.g. `HADES_TRAINER_DEVICE=`) falls back to
        # the default rather than disabling it.
        return cls(
            socket_path=os.environ.get("HADES_TRAINER_SOCKET", "") or cls.socket_path,
            device=os.environ.get("HADES_TRAINER_DEVICE", "") or cls.device,
        )

    def resolve_device(self, requested: str | None) -> str:
        """Pick the device for a run: the client's requested device when given,
        else this config's default."""
        if requested and requested.strip():
            return requested.strip()
        return self.device
