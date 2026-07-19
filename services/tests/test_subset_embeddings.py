"""CPU tests for neighbourhood-bounded inductive embedding inference."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "generated"))

from hades.training import training_pb2  # noqa: E402
from training.config import TrainingConfig  # noqa: E402
from training.server import TrainingServicer  # noqa: E402


def _loaded_servicer(
    architecture: str = "hetero_sage", device: str = "cpu"
) -> TrainingServicer:
    torch.manual_seed(7)
    servicer = TrainingServicer(TrainingConfig())
    servicer.device = torch.device(device)
    servicer.model_config = training_pb2.ModelConfig(
        num_relations=2,
        num_collection_types=2,
        hidden_dim=8,
        embed_dim=4,
        num_bases=2,
        dropout=0.0,
        architecture=architecture,
    )
    servicer.opt_config = training_pb2.OptimizerConfig(
        learning_rate=0.01, weight_decay=5e-4
    )
    servicer._build_model(in_dim=6)
    servicer.x = torch.randn(7, 6, device=device)
    servicer.node_collections = torch.tensor([0, 0, 1, 0, 1, 1, 0], device=device)
    servicer.edge_src = torch.tensor([0, 1, 2, 3, 4, 5, 0, 6], device=device)
    servicer.edge_dst = torch.tensor([1, 2, 3, 4, 5, 6, 4, 5], device=device)
    servicer.edge_type = torch.tensor([0, 1, 0, 1, 0, 1, 1, 0], device=device)
    servicer.model.eval()
    return servicer


def test_subset_matches_full_forward_and_preserves_request_order():
    servicer = _loaded_servicer()
    with torch.no_grad():
        full = servicer._encode()
        subset = servicer._encode_subset([6, 3])
    torch.testing.assert_close(subset, full[[6, 3]])


def test_subset_rejects_duplicates_and_out_of_range_indices():
    servicer = _loaded_servicer()
    with pytest.raises(ValueError, match="duplicates"):
        servicer._encode_subset([2, 2])
    with pytest.raises(ValueError, match="out of range"):
        servicer._encode_subset([7])


def test_subset_requires_inductive_architecture():
    servicer = _loaded_servicer("rgcn")
    with pytest.raises(ValueError, match="hetero_sage"):
        servicer._encode_subset([2])


def test_get_embeddings_rpc_returns_compact_rows_in_request_order():
    servicer = _loaded_servicer()
    request = training_pb2.GetEmbeddingsRequest(node_indices=[6, 3])
    response = asyncio.run(servicer.GetEmbeddings(request, None))

    assert response.num_nodes == 2
    assert response.embed_dim == 4
    actual = torch.frombuffer(bytearray(response.embeddings), dtype=torch.float32).reshape(2, 4)
    with torch.no_grad():
        expected = servicer._encode()[[6, 3]].cpu()
    torch.testing.assert_close(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_checkpoint_subset_matches_full_forward_on_cuda(tmp_path):
    original = _loaded_servicer(device="cuda:0")
    checkpoint = tmp_path / "subset.pt"
    asyncio.run(
        original.Checkpoint(training_pb2.CheckpointRequest(path=str(checkpoint)), None)
    )

    servicer = TrainingServicer(TrainingConfig())
    asyncio.run(
        servicer.LoadCheckpoint(
            training_pb2.LoadCheckpointRequest(path=str(checkpoint), device="cuda:0"),
            None,
        )
    )
    servicer.x = original.x
    servicer.node_collections = original.node_collections
    servicer.edge_src = original.edge_src
    servicer.edge_dst = original.edge_dst
    servicer.edge_type = original.edge_type
    servicer.model.eval()
    with torch.no_grad():
        full = servicer._encode()
        subset = servicer._encode_subset([6, 3])
    torch.testing.assert_close(subset, full[[6, 3]])
