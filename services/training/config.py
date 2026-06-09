"""Training service configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class TrainingConfig:
    """Configuration for the RGCN training service.

    There is deliberately no fallback GPU device. GNN training is infrequent,
    and an infrequently exercised default is a forgotten default: it rots
    between uses and lands a heavy job on the wrong card with nobody having
    chosen it. Every run declares its device in the request (driven by the
    CLI's mandatory ``--gpu`` flag), and the server rejects requests that
    carry no device.
    """

    socket_path: str = "/run/hades/training.sock"

    @classmethod
    def from_env(cls) -> TrainingConfig:
        # `.strip() or default` so a blank or whitespace-only env entry
        # falls back to the default rather than producing an invalid path.
        return cls(
            socket_path=os.environ.get("HADES_TRAINER_SOCKET", "").strip() or cls.socket_path,
        )

    def resolve_device(self, requested: str | None) -> str | None:
        """Return the client's declared device, or None when absent.

        None means the caller must abort the RPC: there is no fallback by
        design (productive friction on a rare, heavy operation).
        """
        if requested and requested.strip():
            return requested.strip()
        return None
