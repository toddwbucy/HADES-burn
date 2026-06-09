"""CPU tests for the architecture selector in the training server.

Verifies that ModelConfig.architecture picks the encoder class — "rgcn" (or the
empty-string back-compat default) builds HadesRGCN, "hetero_sage" builds the
inductive HadesHeteroSAGE — and that an unknown value fails loudly. No GPU/DB.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "generated"))

from hades.training import training_pb2  # noqa: E402
from training.config import TrainingConfig  # noqa: E402
from training.rgcn_model import HadesRGCN  # noqa: E402
from training.sage_model import HadesHeteroSAGE  # noqa: E402
from training.server import TrainingServicer  # noqa: E402


def _servicer(architecture: str) -> TrainingServicer:
    s = TrainingServicer(TrainingConfig())
    s.device = torch.device("cpu")
    s.model_config = training_pb2.ModelConfig(
        num_relations=3,
        num_collection_types=1,
        hidden_dim=12,
        embed_dim=8,
        num_bases=2,
        dropout=0.0,
        architecture=architecture,
    )
    s.opt_config = training_pb2.OptimizerConfig(learning_rate=0.01, weight_decay=5e-4)
    return s


@pytest.mark.parametrize(
    "arch,expected",
    [
        ("", HadesRGCN),            # back-compat default
        ("rgcn", HadesRGCN),
        ("hetero_sage", HadesHeteroSAGE),
    ],
)
def test_build_model_selects_architecture(arch, expected):
    s = _servicer(arch)
    n_params = s._build_model(in_dim=16)
    assert isinstance(s.model, expected)
    assert n_params > 0


def test_unknown_architecture_fails_loud():
    s = _servicer("graphsage_typo")
    with pytest.raises(ValueError, match="unknown architecture"):
        s._build_model(in_dim=16)
