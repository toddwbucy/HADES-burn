"""CPU unit tests for the inductive relational GraphSAGE encoder.

These do not require a GPU or any database — they run on tiny synthetic
multi-relational graphs and pin down the two properties that justify adding
SAGE alongside RGCN: relation-awareness (edge types aggregate separately) and
inductiveness (one frozen model embeds graphs of different size, including
nodes it never saw).

CI is Rust-only, so these are run locally (``make -C services test`` once the
``dev`` extra is installed); they double as executable documentation of the
inductive contract.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

# Import the module under test without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.sage_model import HadesHeteroSAGE  # noqa: E402


def _tiny_graph(n: int, num_relations: int, in_dim: int, seed: int = 0):
    """A small reproducible multi-relational graph: features, types, edges."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, in_dim, generator=g)
    node_collections = torch.zeros(n, dtype=torch.long)
    # A ring of edges, alternating relation ids so >1 relation is exercised.
    src = torch.arange(n)
    dst = (src + 1) % n
    edge_type = (src % num_relations)
    return x, node_collections, src, dst, edge_type


def _model(num_relations=3, num_collection_types=1, in_dim=16, embed_dim=8):
    torch.manual_seed(0)
    return HadesHeteroSAGE(
        num_relations=num_relations,
        num_collection_types=num_collection_types,
        in_dim=in_dim,
        hidden_dim=12,
        embed_dim=embed_dim,
        num_bases=2,
        num_layers=2,
        dropout=0.0,
    ).eval()


def test_encode_shape_and_finite():
    m = _model()
    x, nc, s, d, et = _tiny_graph(n=6, num_relations=3, in_dim=16)
    emb = m.encode(x, nc, s, d, et)
    assert emb.shape == (6, 8)
    assert torch.isfinite(emb).all()


def test_output_is_l2_normalized():
    """Every (non-degenerate) embedding row is unit-norm — the SAGE invariant
    that makes the downstream dot-product equal cosine similarity."""
    m = _model()
    x, nc, s, d, et = _tiny_graph(n=6, num_relations=3, in_dim=16)
    emb = m.encode(x, nc, s, d, et)
    norms = emb.norm(p=2, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_relations_are_distinguished():
    """Relabelling an edge's relation changes the embedding — proof the layer
    does not collapse edge types into one homogeneous neighbourhood."""
    m = _model(num_relations=3)
    x, nc, s, d, et = _tiny_graph(n=6, num_relations=3, in_dim=16)
    emb_a = m.encode(x, nc, s, d, et)
    et2 = et.clone()
    et2[0] = (et2[0] + 1) % 3  # move one edge to a different relation
    emb_b = m.encode(x, nc, s, d, et2)
    assert not torch.allclose(emb_a, emb_b, atol=1e-6)


def test_score_is_dot_product():
    m = _model()
    x, nc, s, d, et = _tiny_graph(n=6, num_relations=3, in_dim=16)
    emb = m.encode(x, nc, s, d, et)
    src = torch.tensor([0, 1, 2])
    dst = torch.tensor([3, 4, 5])
    expected = (emb[src] * emb[dst]).sum(dim=-1)
    assert torch.allclose(m.score(emb, src, dst), expected)


def test_no_per_node_parameters():
    """Inductiveness, structurally: no parameter dimension scales with the node
    count. The constructor never receives N, so a per-node embedding table
    cannot exist — the encoder is a function of features + neighbourhood only."""
    m = _model()
    for n in (5, 50, 500):
        for p in m.parameters():
            assert n not in tuple(p.shape), (
                f"parameter {tuple(p.shape)} scales with node count {n}"
            )


def test_inductive_generalizes_to_unseen_graph_sizes():
    """One frozen model embeds graphs of different N — including a larger graph
    of nodes never seen — producing finite, correctly-shaped embeddings. This
    is the inductive property RGCN-in-deployment lacks."""
    m = _model()
    for n in (4, 9, 23):
        x, nc, s, d, et = _tiny_graph(n=n, num_relations=3, in_dim=16, seed=n)
        emb = m.encode(x, nc, s, d, et)
        assert emb.shape == (n, 8)
        assert torch.isfinite(emb).all()


def test_new_node_embedding_depends_on_its_features():
    """Add a brand-new node to an existing graph and embed with frozen weights:
    its embedding is finite and is a function of *its own* features (perturbing
    them changes it). No retraining, no node-id lookup — a forward pass."""
    m = _model()
    n, in_dim = 6, 16
    x, nc, s, d, et = _tiny_graph(n=n, num_relations=3, in_dim=in_dim)

    # Append node n, wired to existing nodes 0 and 1 (one of each relation).
    new_feat = torch.randn(1, in_dim, generator=torch.Generator().manual_seed(99))
    x_ext = torch.cat([x, new_feat], dim=0)
    nc_ext = torch.cat([nc, torch.zeros(1, dtype=torch.long)])
    s_ext = torch.cat([s, torch.tensor([0, 1])])
    d_ext = torch.cat([d, torch.tensor([n, n])])
    et_ext = torch.cat([et, torch.tensor([0, 1])])

    emb_ext = m.encode(x_ext, nc_ext, s_ext, d_ext, et_ext)
    new_emb = emb_ext[n]
    assert torch.isfinite(new_emb).all()
    assert new_emb.norm() > 0  # not a degenerate zero row

    # Perturb only the new node's features; its embedding must move.
    x_ext2 = x_ext.clone()
    x_ext2[n] = x_ext2[n] + 1.0
    emb_ext2 = m.encode(x_ext2, nc_ext, s_ext, d_ext, et_ext)
    assert not torch.allclose(new_emb, emb_ext2[n], atol=1e-6)


def test_inductive_after_training_step():
    """Sanity that the model is trainable through the shared score head and,
    once weights are frozen, still embeds an unseen node — i.e. training does
    not make it transductive."""
    m = HadesHeteroSAGE(
        num_relations=3, num_collection_types=1, in_dim=16,
        hidden_dim=12, embed_dim=8, num_bases=2, num_layers=2, dropout=0.0,
    ).train()
    x, nc, s, d, et = _tiny_graph(n=8, num_relations=3, in_dim=16)
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)

    neg_dst = (d + 3) % 8  # corrupted tails as negatives
    for _ in range(5):
        opt.zero_grad()
        emb = m.encode(x, nc, s, d, et)
        pos = m.score(emb, s, d)
        neg = m.score(emb, s, neg_dst)
        # BCE on logits: positives -> 1, negatives -> 0.
        loss = (
            torch.nn.functional.binary_cross_entropy_with_logits(pos, torch.ones_like(pos))
            + torch.nn.functional.binary_cross_entropy_with_logits(neg, torch.zeros_like(neg))
        )
        loss.backward()
        opt.step()
    assert torch.isfinite(loss)

    # Freeze and embed a graph one node larger (a node unseen during training).
    m.eval()
    with torch.no_grad():
        x2, nc2, s2, d2, et2 = _tiny_graph(n=9, num_relations=3, in_dim=16, seed=9)
        emb2 = m.encode(x2, nc2, s2, d2, et2)
    assert emb2.shape == (9, 8)
    assert torch.isfinite(emb2).all()
