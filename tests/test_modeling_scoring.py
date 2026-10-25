"""Tests for the score_pair adapters shared by the CLI and multi-seed runner."""

from __future__ import annotations

from pathlib import Path

import torch

from app.modeling.loader import load_graph_package
from app.modeling.models import BaselineScorer, GNNLinkPredictor, compute_gene_prior
from app.modeling.scoring import baseline_score_pair, gnn_score_pair

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "modeling"


def test_baseline_score_pair_matches_direct_forward_call() -> None:
    package = load_graph_package(FIXTURE_DIR, seed=42)
    disease_ids = package.node_indexes["Disease"].ids
    gene_ids = package.node_indexes["Gene"].ids
    gene_prior = compute_gene_prior(package.positives["train"], gene_ids)
    model = BaselineScorer(disease_ids, gene_ids, gene_prior, seed=42)

    pairs = package.positives["validation"][:1]
    score_pair = baseline_score_pair(model, package)
    scores = score_pair(pairs)

    disease_index = package.node_indexes["Disease"].index
    gene_index = package.node_indexes["Gene"].index
    disease_idx = torch.tensor([disease_index[d] for d, _g in pairs], dtype=torch.long)
    gene_idx = torch.tensor([gene_index[g] for _d, g in pairs], dtype=torch.long)
    model.eval()
    with torch.no_grad():
        expected = model(disease_idx, gene_idx).tolist()

    assert scores == expected


def test_gnn_score_pair_matches_direct_decode_call() -> None:
    package = load_graph_package(FIXTURE_DIR, seed=42)
    num_nodes = {
        node_type: len(index.ids) for node_type, index in package.node_indexes.items()
    }
    model = GNNLinkPredictor(
        package.data.metadata(), num_nodes=num_nodes, hidden_dim=8, embedding_dim=8, seed=42
    )

    pairs = package.positives["validation"][:1]
    score_pair = gnn_score_pair(model, package)
    scores = score_pair(pairs)

    disease_index = package.node_indexes["Disease"].index
    gene_index = package.node_indexes["Gene"].index
    disease_idx = torch.tensor([disease_index[d] for d, _g in pairs], dtype=torch.long)
    gene_idx = torch.tensor([gene_index[g] for _d, g in pairs], dtype=torch.long)
    model.eval()
    with torch.no_grad():
        embeddings = model.encode(package.data)
        expected = model.decode(
            embeddings["Disease"], embeddings["Gene"], disease_idx, gene_idx
        ).tolist()

    assert scores == expected


def test_gnn_score_pair_caches_encode_and_does_not_recompute_per_call() -> None:
    package = load_graph_package(FIXTURE_DIR, seed=42)
    num_nodes = {
        node_type: len(index.ids) for node_type, index in package.node_indexes.items()
    }
    model = GNNLinkPredictor(
        package.data.metadata(), num_nodes=num_nodes, hidden_dim=8, embedding_dim=8, seed=42
    )

    encode_calls = []
    original_encode = GNNLinkPredictor.encode

    def counting_encode(self, data):  # type: ignore[no-untyped-def]
        encode_calls.append(1)
        return original_encode(self, data)

    GNNLinkPredictor.encode = counting_encode  # type: ignore[method-assign]
    try:
        score_pair = gnn_score_pair(model, package)
        assert len(encode_calls) == 1
        score_pair(package.positives["validation"][:1])
        score_pair(package.positives["test"][:1])
        assert len(encode_calls) == 1
    finally:
        GNNLinkPredictor.encode = original_encode  # type: ignore[method-assign]
