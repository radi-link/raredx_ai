"""Adapt `BaselineScorer`/`GNNLinkPredictor` to the `evaluate_link_prediction`
`score_pair` callable contract: `Callable[[list[(disease_id, gene_id)]], list[float]]`.
"""

from __future__ import annotations

from typing import Callable

import torch

from app.modeling.loader import GraphPackage
from app.modeling.models import BaselineScorer, GNNLinkPredictor

Pair = tuple[str, str]
ScorePairFn = Callable[[list[Pair]], list[float]]


def pairs_to_index_tensors(
    pairs: list[Pair], disease_index: dict[str, int], gene_index: dict[str, int]
) -> tuple[torch.LongTensor, torch.LongTensor]:
    """Convert (disease_id, gene_id) pairs to parallel LongTensors of positions.

    Shared by this module's adapters and `app/modeling/train.py`'s batch building
    so the two don't carry independent, driftable copies of the same conversion.
    """
    disease_idx = torch.tensor(
        [disease_index[disease_id] for disease_id, _gene_id in pairs], dtype=torch.long
    )
    gene_idx = torch.tensor(
        [gene_index[gene_id] for _disease_id, gene_id in pairs], dtype=torch.long
    )
    return disease_idx, gene_idx


def baseline_score_pair(model: BaselineScorer, package: GraphPackage) -> ScorePairFn:
    """Adapt `BaselineScorer` to the `score_pair` callable contract."""
    disease_index = package.node_indexes["Disease"].index
    gene_index = package.node_indexes["Gene"].index

    model.eval()

    def score_pair(pairs: list[Pair]) -> list[float]:
        disease_idx, gene_idx = pairs_to_index_tensors(pairs, disease_index, gene_index)
        with torch.no_grad():
            logits = model(disease_idx, gene_idx)
        return logits.tolist()

    return score_pair


def gnn_score_pair(model: GNNLinkPredictor, package: GraphPackage) -> ScorePairFn:
    """Adapt `GNNLinkPredictor` to the `score_pair` callable contract.

    Encodes the graph once up front (eval mode, no grad) and caches the resulting
    Disease/Gene embeddings, so repeated `score_pair` calls during evaluation only
    run the cheap decode, not the full GNN forward pass.
    """
    disease_index = package.node_indexes["Disease"].index
    gene_index = package.node_indexes["Gene"].index

    model.eval()
    with torch.no_grad():
        embeddings = model.encode(package.data)
    disease_emb = embeddings["Disease"]
    gene_emb = embeddings["Gene"]

    def score_pair(pairs: list[Pair]) -> list[float]:
        disease_idx, gene_idx = pairs_to_index_tensors(pairs, disease_index, gene_index)
        with torch.no_grad():
            logits = model.decode(disease_emb, gene_emb, disease_idx, gene_idx)
        return logits.tolist()

    return score_pair
