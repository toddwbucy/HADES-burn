"""HADES RGCN training service — gRPC server (Python GPU process).

The Rust orchestrator drives the training loop and issues per-step RPCs; this
server owns the GPU, model parameters, and optimizer state. It implements the
`hades.training.TrainingService` lifecycle:

    InitModel -> LoadGraph -> [TrainStep* / Evaluate]* -> GetEmbeddings
                                                       -> Checkpoint/LoadCheckpoint

HADES-owned compute service (not the Persephone PM system); decoupled from the
`persephone.*` provider brand (issue #106). Listens on a Unix socket
(`/run/hades/training.sock` by default), mirroring the embedder/extractor.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import stat
import struct
import sys
from pathlib import Path

import grpc
import torch
from grpc import aio as grpc_aio

# Ensure generated stubs are importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "generated"))

from hades.training import training_pb2, training_pb2_grpc  # noqa: E402

from .config import TrainingConfig  # noqa: E402
from .rgcn_model import HadesRGCN  # noqa: E402

logger = logging.getLogger("hades.training")

# Node feature width when the client doesn't tell us otherwise — Jina V4 width,
# matching the schema_meta `feature_dim` default. LoadGraph rebuilds if the
# actual graph differs.
DEFAULT_FEATURE_DIM = 2048


def _bce_link_loss(pos_score: torch.Tensor, neg_score: torch.Tensor):
    """Binary cross-entropy link loss + accuracy over positives/negatives."""
    pos_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        pos_score, torch.ones_like(pos_score)
    )
    neg_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        neg_score, torch.zeros_like(neg_score)
    )
    loss = pos_loss + neg_loss
    with torch.no_grad():
        pos_acc = (pos_score > 0).float().mean() if pos_score.numel() else torch.tensor(0.0)
        neg_acc = (neg_score <= 0).float().mean() if neg_score.numel() else torch.tensor(0.0)
        acc = 0.5 * (pos_acc + neg_acc)
    return loss, float(acc)


def _auc(pos_score: torch.Tensor, neg_score: torch.Tensor) -> float:
    """ROC-AUC via the Mann–Whitney U statistic (no sklearn dependency)."""
    n_pos, n_neg = pos_score.numel(), neg_score.numel()
    if n_pos == 0 or n_neg == 0:
        return 0.0
    scores = torch.cat([pos_score, neg_score])
    ranks = scores.argsort().argsort().float() + 1.0  # average-rank approx
    rank_pos = ranks[:n_pos].sum()
    auc = (rank_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc.clamp(0.0, 1.0))


class TrainingServicer(training_pb2_grpc.TrainingServiceServicer):
    def __init__(self, config: TrainingConfig) -> None:
        self.config = config
        self.device: torch.device | None = None
        self.model: HadesRGCN | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.model_config = None
        self.opt_config = None
        self.in_dim = DEFAULT_FEATURE_DIM
        # Graph tensors (set by LoadGraph)
        self.x = None
        self.node_collections = None
        self.edge_src = None
        self.edge_dst = None
        self.edge_type = None

    # -- helpers ----------------------------------------------------------
    def _build_model(self, in_dim: int) -> int:
        mc = self.model_config
        self.in_dim = in_dim
        self.model = HadesRGCN(
            num_relations=mc.num_relations,
            num_collection_types=mc.num_collection_types,
            in_dim=in_dim,
            hidden_dim=mc.hidden_dim or 256,
            embed_dim=mc.embed_dim or 128,
            num_bases=mc.num_bases or mc.num_relations,
            dropout=mc.dropout,
        ).to(self.device)
        oc = self.opt_config
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=oc.learning_rate or 0.01,
            weight_decay=oc.weight_decay or 5e-4,
        )
        return sum(p.numel() for p in self.model.parameters())

    def _require_loaded(self, context):
        if self.model is None:
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, "InitModel not called")
        if self.x is None:
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, "LoadGraph not called")

    def _idx(self, values) -> torch.Tensor:
        return torch.as_tensor(list(values), dtype=torch.long, device=self.device)

    # -- RPCs -------------------------------------------------------------
    async def InitModel(self, request, context):
        device = self.config.resolve_device(request.device)
        self.device = torch.device(device)
        self.model_config = request.model
        self.opt_config = request.optimizer
        num_params = self._build_model(DEFAULT_FEATURE_DIM)
        logger.info("InitModel: %d params on %s", num_params, device)
        return training_pb2.InitModelResponse(num_parameters=num_params, device=device)

    async def LoadGraph(self, request, context):
        if self.model is None:
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, "InitModel not called")
        from safetensors.torch import load_file

        t = load_file(request.safetensors_path)
        x = t["node_features"].to(self.device).float()
        feature_dim = x.size(1)
        if feature_dim != self.in_dim:
            logger.info("rebuilding model for feature_dim=%d (was %d)", feature_dim, self.in_dim)
            self._build_model(feature_dim)

        self.x = x
        self.node_collections = t["node_collections"].to(self.device).long()
        self.edge_src = t["edge_src"].to(self.device).long()
        self.edge_dst = t["edge_dst"].to(self.device).long()
        self.edge_type = t["edge_type"].to(self.device).long()

        num_nodes = self.x.size(0)
        num_edges = self.edge_src.size(0)
        gpu_bytes = 0
        if self.device.type == "cuda":
            gpu_bytes = torch.cuda.memory_allocated(self.device)
        logger.info("LoadGraph: %d nodes, %d edges, dim=%d", num_nodes, num_edges, feature_dim)
        return training_pb2.LoadGraphResponse(
            num_nodes=num_nodes,
            num_edges=num_edges,
            feature_dim=feature_dim,
            gpu_memory_bytes=gpu_bytes,
        )

    def _encode(self) -> torch.Tensor:
        return self.model.encode(
            self.x, self.node_collections, self.edge_src, self.edge_dst, self.edge_type
        )

    async def TrainStep(self, request, context):
        self._require_loaded(context)
        self.model.train()
        self.optimizer.zero_grad()
        emb = self._encode()
        pos_idx = self._idx(request.train_edge_indices)
        pos_src, pos_dst = self.edge_src[pos_idx], self.edge_dst[pos_idx]
        neg_src, neg_dst = self._idx(request.neg_src), self._idx(request.neg_dst)
        pos_score = HadesRGCN.score(emb, pos_src, pos_dst)
        neg_score = HadesRGCN.score(emb, neg_src, neg_dst)
        loss, acc = _bce_link_loss(pos_score, neg_score)
        loss.backward()
        self.optimizer.step()
        return training_pb2.TrainStepResponse(loss=float(loss.detach()), accuracy=acc)

    async def Evaluate(self, request, context):
        self._require_loaded(context)
        self.model.eval()
        with torch.no_grad():
            emb = self._encode()
            pos_idx = self._idx(request.edge_indices)
            pos_src, pos_dst = self.edge_src[pos_idx], self.edge_dst[pos_idx]
            neg_src, neg_dst = self._idx(request.neg_src), self._idx(request.neg_dst)
            pos_score = HadesRGCN.score(emb, pos_src, pos_dst)
            neg_score = HadesRGCN.score(emb, neg_src, neg_dst)
            loss, acc = _bce_link_loss(pos_score, neg_score)
            auc = _auc(pos_score, neg_score)
        return training_pb2.EvaluateResponse(loss=float(loss), accuracy=acc, auc=auc)

    async def GetEmbeddings(self, request, context):
        self._require_loaded(context)
        self.model.eval()
        with torch.no_grad():
            emb = self._encode().cpu().contiguous().float()
        num_nodes, embed_dim = emb.size(0), emb.size(1)
        if request.output_path:
            from safetensors.torch import save_file

            save_file({"structural_embedding": emb}, request.output_path)
            return training_pb2.GetEmbeddingsResponse(
                num_nodes=num_nodes, embed_dim=embed_dim, output_path=request.output_path
            )
        payload = emb.numpy().tobytes()
        return training_pb2.GetEmbeddingsResponse(
            num_nodes=num_nodes, embed_dim=embed_dim, embeddings=payload
        )

    async def Checkpoint(self, request, context):
        if self.model is None:
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, "InitModel not called")
        Path(request.path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "in_dim": self.in_dim,
                "model_config": {
                    "num_relations": self.model_config.num_relations,
                    "num_collection_types": self.model_config.num_collection_types,
                    "hidden_dim": self.model_config.hidden_dim,
                    "embed_dim": self.model_config.embed_dim,
                    "num_bases": self.model_config.num_bases,
                    "dropout": self.model_config.dropout,
                },
            },
            request.path,
        )
        size = os.path.getsize(request.path)
        return training_pb2.CheckpointResponse(path=request.path, size_bytes=size)

    async def LoadCheckpoint(self, request, context):
        device = self.config.resolve_device(request.device)
        self.device = torch.device(device)
        # weights_only=True: the checkpoint holds only tensors + basic types,
        # and refusing arbitrary unpickling avoids code execution from a
        # tampered checkpoint file.
        ckpt = torch.load(request.path, map_location=self.device, weights_only=True)
        mc = ckpt["model_config"]
        self.model_config = training_pb2.ModelConfig(**mc)
        self.opt_config = self.opt_config or training_pb2.OptimizerConfig()
        num_params = self._build_model(ckpt.get("in_dim", DEFAULT_FEATURE_DIM))
        self.model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        return training_pb2.LoadCheckpointResponse(
            model_config=self.model_config, num_parameters=num_params, device=device
        )


async def serve() -> None:
    config = TrainingConfig.from_env()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    servicer = TrainingServicer(config)
    server = grpc_aio.server(options=[
        ("grpc.max_send_message_length", 512 * 1024 * 1024),
        ("grpc.max_receive_message_length", 512 * 1024 * 1024),
    ])
    training_pb2_grpc.add_TrainingServiceServicer_to_server(servicer, server)

    socket_path = config.socket_path
    Path(socket_path).parent.mkdir(parents=True, exist_ok=True)
    sock = Path(socket_path)
    if sock.exists():
        if stat.S_ISSOCK(sock.stat().st_mode):
            sock.unlink()
        else:
            raise RuntimeError(f"Path {socket_path} exists but is not a socket")

    server.add_insecure_port(f"unix:{socket_path}")
    logger.info("Starting training service on %s (default device %s)", socket_path, config.device)
    await server.start()

    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stop_event.set)

    logger.info("Training service ready")
    await stop_event.wait()

    logger.info("Shutting down training service...")
    await server.stop(grace=5)
    if sock.exists() and stat.S_ISSOCK(sock.stat().st_mode):
        sock.unlink()
    logger.info("Training service stopped")


if __name__ == "__main__":
    asyncio.run(serve())
