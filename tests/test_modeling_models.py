"""Tests for the baseline scorer and the heterogeneous GNN link predictor."""

from __future__ import annotations

from pathlib import Path

import torch
from torch_geometric.data import HeteroData

from app.modeling.loader import GraphPackage, load_graph_package
from app.modeling.models import BaselineScorer, GNNLinkPredictor, compute_gene_prior

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "modeling"


def test_compute_gene_prior_normalizes_and_defaults_to_zero() -> None:
    prior = compute_gene_prior(
        [("ORPHA:1", "HGNC:1"), ("ORPHA:2", "HGNC:1")], ["HGNC:1", "HGNC:2"]
    )
    assert prior["HGNC:1"] == 1.0
    assert prior["HGNC:2"] == 0.0


def test_compute_gene_prior_normalizes_by_max_count() -> None:
    prior = compute_gene_prior(
        [
            ("ORPHA:1", "HGNC:1"),
            ("ORPHA:2", "HGNC:1"),
            ("ORPHA:3", "HGNC:2"),
        ],
        ["HGNC:1", "HGNC:2", "HGNC:3"],
    )
    assert prior["HGNC:1"] == 1.0
    assert prior["HGNC:2"] == 0.5
    assert prior["HGNC:3"] == 0.0


def test_compute_gene_prior_handles_empty_positives() -> None:
    prior = compute_gene_prior([], ["HGNC:1", "HGNC:2"])
    assert prior["HGNC:1"] == 0.0
    assert prior["HGNC:2"] == 0.0


def test_baseline_scorer_forward_returns_one_logit_per_pair() -> None:
    torch.manual_seed(0)
    model = BaselineScorer(
        ["ORPHA:1", "ORPHA:2"], ["HGNC:1", "HGNC:2"], {"HGNC:1": 1.0, "HGNC:2": 0.0}
    )
    scores = model(torch.tensor([0, 1]), torch.tensor([1, 0]))
    assert scores.shape == (2,)
    assert torch.isfinite(scores).all()


def test_baseline_scorer_is_deterministic_for_fixed_seed() -> None:
    model_a = BaselineScorer(["ORPHA:1"], ["HGNC:1"], {"HGNC:1": 0.5}, seed=42)
    model_b = BaselineScorer(["ORPHA:1"], ["HGNC:1"], {"HGNC:1": 0.5}, seed=42)
    assert torch.equal(model_a.disease_embedding.weight, model_b.disease_embedding.weight)


def test_baseline_scorer_different_seeds_differ() -> None:
    model_a = BaselineScorer(["ORPHA:1"], ["HGNC:1"], {"HGNC:1": 0.5}, seed=1)
    model_b = BaselineScorer(["ORPHA:1"], ["HGNC:1"], {"HGNC:1": 0.5}, seed=2)
    assert not torch.equal(model_a.disease_embedding.weight, model_b.disease_embedding.weight)


def test_baseline_scorer_does_not_touch_global_rng_state() -> None:
    torch.manual_seed(123)
    before = torch.rand(3)

    torch.manual_seed(123)
    BaselineScorer(["ORPHA:1"], ["HGNC:1"], {"HGNC:1": 0.5}, seed=999)
    after = torch.rand(3)

    assert torch.equal(before, after)


def test_baseline_scorer_gene_prior_logit_contributes_to_unseen_gene() -> None:
    torch.manual_seed(0)
    model = BaselineScorer(
        ["ORPHA:1"], ["HGNC:1", "HGNC:2"], {"HGNC:1": 1.0, "HGNC:2": 0.0}
    )
    # unseen gene (HGNC:2, prior 0.0) should not get an automatic prior boost
    assert model.gene_prior[1].item() == 0.0
    assert model.gene_prior[0].item() == 1.0


def test_baseline_scorer_embedding_dim_default() -> None:
    model = BaselineScorer(["ORPHA:1"], ["HGNC:1"], {"HGNC:1": 0.5})
    assert model.disease_embedding.embedding_dim == 32
    assert model.gene_embedding.embedding_dim == 32


def _num_nodes(package: GraphPackage) -> dict[str, int]:
    return {node_type: len(index.ids) for node_type, index in package.node_indexes.items()}


def _build_gnn(package: GraphPackage, *, seed: int = 42) -> GNNLinkPredictor:
    return GNNLinkPredictor(
        package.data.metadata(),
        num_nodes=_num_nodes(package),
        hidden_dim=8,
        embedding_dim=8,
        seed=seed,
    )


def test_gnn_encode_returns_embedding_per_node() -> None:
    package = load_graph_package(FIXTURE_DIR, seed=42)
    model = _build_gnn(package)
    embeddings = model.encode(package.data)
    assert embeddings["Disease"].shape == (len(package.node_indexes["Disease"].ids), 8)
    assert embeddings["Gene"].shape == (len(package.node_indexes["Gene"].ids), 8)
    assert torch.isfinite(embeddings["Disease"]).all()
    assert torch.isfinite(embeddings["Gene"]).all()


def test_gnn_decode_returns_one_logit_per_pair() -> None:
    package = load_graph_package(FIXTURE_DIR, seed=42)
    model = _build_gnn(package)
    embeddings = model.encode(package.data)
    scores = model.decode(
        embeddings["Disease"], embeddings["Gene"], torch.tensor([0]), torch.tensor([0])
    )
    assert scores.shape == (1,)
    assert torch.isfinite(scores).all()


def test_gnn_decode_batch_shape_matches_baseline_contract() -> None:
    package = load_graph_package(FIXTURE_DIR, seed=42)
    model = _build_gnn(package)
    embeddings = model.encode(package.data)
    scores = model.decode(
        embeddings["Disease"],
        embeddings["Gene"],
        torch.tensor([0, 1, 2]),
        torch.tensor([2, 1, 0]),
    )
    assert scores.shape == (3,)


def test_gnn_convs_cover_every_edge_type_including_reverse() -> None:
    package = load_graph_package(FIXTURE_DIR, seed=42)
    model = _build_gnn(package)
    edge_types = set(package.data.metadata()[1])
    assert any(relation.startswith("rev_") for _, relation, _ in edge_types)
    for conv in model.convs:
        assert set(conv.convs.keys()) == edge_types


def test_gnn_disease_embeddings_depend_on_reverse_edges() -> None:
    # Disease nodes only receive messages over rev_* edge types, so dropping them
    # must change the Disease output if encode() really uses reverse edges.
    package = load_graph_package(FIXTURE_DIR, seed=42)
    model = _build_gnn(package)
    model.eval()

    forward_only = HeteroData()
    for node_type in package.data.node_types:
        forward_only[node_type].num_nodes = package.data[node_type].num_nodes
    for edge_type in package.data.edge_types:
        if not edge_type[1].startswith("rev_"):
            forward_only[edge_type].edge_index = package.data[edge_type].edge_index

    with torch.no_grad():
        full = model.encode(package.data)["Disease"]
        without_reverse = model.encode(forward_only)["Disease"]
    assert not torch.allclose(full, without_reverse)


def test_gnn_is_deterministic_for_fixed_seed() -> None:
    package = load_graph_package(FIXTURE_DIR, seed=42)
    model_a = _build_gnn(package, seed=42)
    model_b = _build_gnn(package, seed=42)
    for (name_a, param_a), (name_b, param_b) in zip(
        model_a.named_parameters(), model_b.named_parameters(), strict=True
    ):
        assert name_a == name_b
        assert torch.equal(param_a, param_b), name_a


def test_gnn_different_seeds_differ() -> None:
    package = load_graph_package(FIXTURE_DIR, seed=42)
    model_a = _build_gnn(package, seed=1)
    model_b = _build_gnn(package, seed=2)
    assert not torch.equal(model_a.relation_weight, model_b.relation_weight)


def test_gnn_does_not_touch_global_rng_state() -> None:
    package = load_graph_package(FIXTURE_DIR, seed=42)

    torch.manual_seed(123)
    before = torch.rand(3)

    torch.manual_seed(123)
    _build_gnn(package, seed=999)
    after = torch.rand(3)

    assert torch.equal(before, after)


def test_gnn_encode_in_train_mode_does_not_touch_global_rng_state() -> None:
    package = load_graph_package(FIXTURE_DIR, seed=42)
    model = _build_gnn(package, seed=999)
    model.train()

    torch.manual_seed(123)
    before = torch.rand(3)

    torch.manual_seed(123)
    model.encode(package.data)
    after = torch.rand(3)

    assert torch.equal(before, after)
