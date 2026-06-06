"""Relational, inductive GraphSAGE for HADES structural graph embeddings.

A sibling of :class:`rgcn_model.HadesRGCN` for the *inductive* situation: when
the graph grows continuously (new code ingested, new symbols, new commits) and
we need to embed freshly-added nodes **without retraining the whole graph**.

RGCN (the transductive option) and this model share one tensor contract — the
flat layout the Rust orchestrator serialises to safetensors:

    node_features    F32  [N, in_dim]
    node_collections U32  [N]            (per-node vertex-type id)
    edge_src/dst     U32  [E]
    edge_type        U32  [E]            (relation id per edge)

and one consumption contract: ``encode(...) -> [N, embed_dim]`` plus a static
relation-agnostic ``score`` (dot product) for link prediction. They are
interchangeable behind the training loop; the schema picks which to use.

Why this one is *inductive* (where RGCN-in-deployment is not)
------------------------------------------------------------
The defining property is structural, not incidental: this module has **no
per-node parameter table**. Its constructor never sees the node count ``N`` —
every weight is shared across nodes and applied to *input features* and a
*sampled neighbourhood*. So the same frozen weights embed a node that did not
exist at training time, by a forward pass over its features + neighbours. The
unit tests prove exactly this: one model, run over graphs of different ``N``,
including brand-new nodes.

The GraphSAGE-mean recipe (vs RGCN's degree-normalised sum + additive self-loop):
  - per-relation **mean** aggregator — invariant to neighbourhood size, so a
    new node of any degree slots in without retraining;
  - **concat** of the node's own transform with its aggregated neighbourhood,
    then a learned combine projection (the SAGE signature);
  - **L2 normalisation** of every layer's output — the canonical GraphSAGE
    norm, which here doubles as the downstream contract: unit-norm embeddings
    make the consumer's dot-product ranking equal to cosine similarity.

Relation-awareness is preserved (this is *relational* SAGE, not homogeneous):
per-relation neighbour transforms are basis-decomposed exactly as in
``RGCNLayer`` (``num_bases`` basis matrices regularise the ``num_relations``
relation weights), so distinct edge types — ``basis`` vs ``validated_against``
vs ``code↔spec`` — aggregate separately rather than being collapsed into one
homogeneous neighbourhood.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SAGELayer(nn.Module):
    """One relational GraphSAGE-mean convolution (inductive).

    For each relation ``r`` the layer mean-aggregates the *input* features of
    ``r``-neighbours into each destination, transforms the per-relation mean by
    a basis-decomposed weight ``W_neigh[r]``, and sums across relations. The
    aggregated neighbourhood is concatenated with the node's own transform and
    projected by ``combine``. Mean-then-transform is equivalent to
    transform-then-mean for a linear map, and cheaper.

    Output activation / normalisation is applied by the encoder (so the final
    layer can skip the nonlinearity while every layer is L2-normalised).
    """

    def __init__(self, in_dim: int, out_dim: int, num_relations: int, num_bases: int):
        super().__init__()
        self.num_relations = num_relations
        self.num_bases = max(1, min(num_bases, num_relations))
        # Basis decomposition: W_neigh[r] = sum_b coeff[r, b] * basis[b].
        # Keeps params at num_bases*in*out instead of num_relations*in*out,
        # mirroring RGCNLayer's regularisation.
        self.basis = nn.Parameter(torch.empty(self.num_bases, in_dim, out_dim))
        self.coeff = nn.Parameter(torch.empty(num_relations, self.num_bases))
        self.self_lin = nn.Linear(in_dim, out_dim, bias=False)
        self.combine = nn.Linear(2 * out_dim, out_dim)
        nn.init.xavier_uniform_(self.basis)
        nn.init.xavier_uniform_(self.coeff)

    def forward(
        self,
        x: torch.Tensor,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        edge_type: torch.Tensor,
    ) -> torch.Tensor:
        weight = torch.einsum("rb,bio->rio", self.coeff, self.basis)  # [R, in, out]
        out_dim = weight.size(2)

        neigh = torch.zeros(x.size(0), out_dim, device=x.device, dtype=x.dtype)
        for r in range(self.num_relations):
            mask = edge_type == r
            if not bool(mask.any()):
                continue
            src_r = edge_src[mask]
            dst_r = edge_dst[mask]
            # Mean-aggregate raw neighbour features per destination for relation r.
            agg_r = torch.zeros(x.size(0), x.size(1), device=x.device, dtype=x.dtype)
            agg_r.index_add_(0, dst_r, x[src_r])
            deg_r = torch.zeros(x.size(0), device=x.device, dtype=x.dtype)
            deg_r.index_add_(0, dst_r, torch.ones(dst_r.size(0), device=x.device, dtype=x.dtype))
            agg_r = agg_r / deg_r.clamp(min=1.0).unsqueeze(1)  # [N, in] — mean
            neigh = neigh + agg_r @ weight[r]  # [N, out]

        h_self = self.self_lin(x)
        return self.combine(torch.cat([h_self, neigh], dim=-1))  # [N, out]


class HadesHeteroSAGE(nn.Module):
    """Inductive relation-aware encoder producing per-node structural embeddings.

    Drop-in sibling of :class:`HadesRGCN`: identical ``__init__`` surface,
    identical ``encode`` signature, identical static ``score``. ``num_bases``
    regularises the per-relation neighbour transforms (as in RGCN); there is no
    per-node parameter, which is what makes the encoder inductive.
    """

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
        cur = hidden_dim
        for i in range(num_layers):
            out_dim = embed_dim if i == num_layers - 1 else hidden_dim
            self.convs.append(SAGELayer(cur, out_dim, num_relations, num_bases))
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

        last = len(self.convs) - 1
        for i, conv in enumerate(self.convs):
            h = conv(h, edge_src, edge_dst, edge_type)
            if i < last:
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
            # Canonical GraphSAGE L2 normalisation, every layer. F.normalize's
            # eps guard leaves all-zero rows at zero (no NaN). The final, unit-
            # norm embedding makes the downstream dot-product == cosine.
            h = F.normalize(h, p=2, dim=-1)
        return h  # [N, embed_dim]

    @staticmethod
    def score(emb: torch.Tensor, src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
        """Link score = dot product of endpoint embeddings (logits).

        Identical to ``HadesRGCN.score`` so the training/eval loop is shared.
        With L2-normalised embeddings this is cosine similarity.
        """
        return (emb[src] * emb[dst]).sum(dim=-1)
