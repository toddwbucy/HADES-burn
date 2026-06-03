"""Homogeneous RGCN for HADES structural graph embeddings.

A self-contained Relational GCN (no torch_geometric dependency) matching the
flat tensor contract the Rust orchestrator serialises to safetensors:

    node_features    F32  [N, in_dim]
    node_collections U32  [N]            (per-node vertex-type id)
    edge_src/dst     U32  [E]
    edge_type        U32  [E]            (relation id per edge)

Architecture (mirrors the Acheron HeteroRGCN, restructured for flat tensors):
  - Per-collection-type input projection: in_dim -> hidden_dim (no bias)
  - `num_layers` RGCNConv layers with basis decomposition (regularises the
    `num_relations` relation matrices), LayerNorm + ReLU + Dropout between
  - Final layer projects to `embed_dim`

Link prediction uses a relation-agnostic dot product, consistent with how the
exported `structural_embedding` is consumed downstream (graph_embed.neighbors
ranks by dot-product similarity).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RGCNLayer(nn.Module):
    """One RGCN convolution with basis decomposition.

    Per-relation weight ``W[r] = sum_b coeff[r, b] * basis[b]`` keeps the
    parameter count at ``num_bases * in * out`` instead of ``num_relations *
    in * out``. Messages are aggregated per relation (a loop over the small
    relation count) to avoid materialising a ``[E, in, out]`` tensor.
    """

    def __init__(self, in_dim: int, out_dim: int, num_relations: int, num_bases: int):
        super().__init__()
        self.num_relations = num_relations
        self.num_bases = max(1, min(num_bases, num_relations))
        self.basis = nn.Parameter(torch.empty(self.num_bases, in_dim, out_dim))
        self.coeff = nn.Parameter(torch.empty(num_relations, self.num_bases))
        self.self_loop = nn.Linear(in_dim, out_dim, bias=False)
        nn.init.xavier_uniform_(self.basis)
        nn.init.xavier_uniform_(self.coeff)

    def forward(
        self,
        x: torch.Tensor,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        edge_type: torch.Tensor,
    ) -> torch.Tensor:
        # Relation weights: [num_relations, in, out]
        weight = torch.einsum("rb,bio->rio", self.coeff, self.basis)
        out_dim = weight.size(2)

        agg = torch.zeros(x.size(0), out_dim, device=x.device, dtype=x.dtype)
        deg = torch.zeros(x.size(0), device=x.device, dtype=x.dtype)

        for r in range(self.num_relations):
            mask = edge_type == r
            if not bool(mask.any()):
                continue
            src_r = edge_src[mask]
            dst_r = edge_dst[mask]
            msg = x[src_r] @ weight[r]  # [E_r, out]
            agg.index_add_(0, dst_r, msg)
            deg.index_add_(0, dst_r, torch.ones(dst_r.size(0), device=x.device, dtype=x.dtype))

        deg = deg.clamp(min=1.0).unsqueeze(1)
        return agg / deg + self.self_loop(x)


class HadesRGCN(nn.Module):
    """Relation-aware encoder producing per-node structural embeddings."""

    def __init__(
        self,
        num_relations: int,
        num_collection_types: int,
        in_dim: int = 2048,
        hidden_dim: int = 256,
        embed_dim: int = 128,
        num_bases: int = 21,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.num_relations = num_relations
        self.num_collection_types = max(1, num_collection_types)
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.dropout = dropout

        # Per-collection-type input projection into the shared hidden space.
        self.projections = nn.ModuleList(
            [nn.Linear(in_dim, hidden_dim, bias=False) for _ in range(self.num_collection_types)]
        )

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        cur = hidden_dim
        for i in range(num_layers):
            out_dim = embed_dim if i == num_layers - 1 else hidden_dim
            self.convs.append(RGCNLayer(cur, out_dim, num_relations, num_bases))
            self.norms.append(nn.LayerNorm(out_dim))
            cur = out_dim

    def encode(
        self,
        x: torch.Tensor,
        node_collections: torch.Tensor,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        edge_type: torch.Tensor,
    ) -> torch.Tensor:
        # Per-type projection. Collection ids outside the known range fall back
        # to projection 0 so an unexpected id can't crash a run.
        h = torch.zeros(x.size(0), self.hidden_dim, device=x.device, dtype=x.dtype)
        for t, proj in enumerate(self.projections):
            mask = node_collections == t
            if bool(mask.any()):
                h[mask] = proj(x[mask])
        unknown = node_collections >= self.num_collection_types
        if bool(unknown.any()):
            h[unknown] = self.projections[0](x[unknown])

        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            h = conv(h, edge_src, edge_dst, edge_type)
            h = norm(h)
            if i < len(self.convs) - 1:
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
        return h  # [N, embed_dim]

    @staticmethod
    def score(emb: torch.Tensor, src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
        """Link score = dot product of endpoint embeddings (logits)."""
        return (emb[src] * emb[dst]).sum(dim=-1)
