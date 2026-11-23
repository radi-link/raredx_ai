"""Tests for the training loops for the baseline and GNN link predictors."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch_geometric.data import HeteroData

from app.modeling.loader import GraphPackage, load_graph_package
from app.modeling.models import GNNLinkPredictor
from app.modeling.train import (
    DG_EDGE_TYPE,
    DG_REVERSE_EDGE_TYPE,
    build_gnn_epoch_batch,
    remove_dg_edges,
    split_message_supervision,
    train_baseline,
    train_gnn,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "modeling"


def test_train_baseline_reduces_loss_on_fixture_graph() -> None:
    package = load_graph_package(FIXTURE_DIR, seed=42)
    _, history = train_baseline(package, epochs=20, seed=42)
    assert history.train_loss[-1] < history.train_loss[0]


def test_train_baseline_is_deterministic_for_fixed_seed() -> None:
    package = load_graph_package(FIXTURE_DIR, seed=42)
    _, history_a = train_baseline(package, epochs=5, seed=42)
    _, history_b = train_baseline(package, epochs=5, seed=42)
    assert history_a.train_loss == history_b.train_loss


def test_train_baseline_records_validation_loss_without_backprop() -> None:
    package = load_graph_package(FIXTURE_DIR, seed=42)
    _, history = train_baseline(package, epochs=5, seed=42)
    assert len(history.validation_loss) == 5
    assert len(history.train_loss) == 5


def test_train_baseline_without_patience_runs_full_epochs_and_tracks_best_epoch() -> None:
    package = load_graph_package(FIXTURE_DIR, seed=42)
    _, history = train_baseline(package, epochs=20, seed=42)
    assert len(history.train_loss) == 20
    assert 1 <= history.best_epoch <= 20
    assert history.validation_loss[history.best_epoch - 1] == min(history.validation_loss)


def test_train_baseline_with_patience_stops_early_and_restores_best_weights() -> None:
    package = load_graph_package(FIXTURE_DIR, seed=42)
    model, history = train_baseline(package, epochs=200, seed=42, patience=5)
    # Validation loss on this tiny fixture overfits quickly, so patience should
    # trigger well before the full epoch budget.
    assert len(history.train_loss) < 200
    assert len(history.train_loss) == history.best_epoch + 5
    assert history.validation_loss[history.best_epoch - 1] == min(history.validation_loss)

    # The restored model's validation loss matches the recorded best, not the
    # loss from the last (post-patience) epoch trained.
    disease_index = package.node_indexes["Disease"].index
    gene_index = package.node_indexes["Gene"].index
    val_pairs = package.positives["validation"] + package.negatives["validation"]
    disease_idx = torch.tensor([disease_index[d] for d, _g in val_pairs], dtype=torch.long)
    gene_idx = torch.tensor([gene_index[g] for _d, g in val_pairs], dtype=torch.long)
    labels = torch.cat(
        [
            torch.ones(len(package.positives["validation"])),
            torch.zeros(len(package.negatives["validation"])),
        ]
    )
    model.eval()
    with torch.no_grad():
        logits = model(disease_idx, gene_idx)
    restored_loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels).item()
    assert restored_loss == pytest.approx(min(history.validation_loss), abs=1e-5)


def test_train_baseline_raises_on_empty_validation_split() -> None:
    package = load_graph_package(FIXTURE_DIR, seed=42)
    package.positives["validation"] = []
    package.negatives["validation"] = []
    with pytest.raises(ValueError, match="validation"):
        train_baseline(package, epochs=2, seed=42)


def test_train_gnn_raises_on_empty_validation_split() -> None:
    package = load_graph_package(FIXTURE_DIR, seed=42)
    package.positives["validation"] = []
    package.negatives["validation"] = []
    with pytest.raises(ValueError, match="validation"):
        train_gnn(package, epochs=2, hidden_dim=8, embedding_dim=8, seed=42)


def test_train_baseline_weight_decay_shrinks_relation_weight() -> None:
    package = load_graph_package(FIXTURE_DIR, seed=42)
    plain, _ = train_baseline(package, epochs=20, seed=42, weight_decay=0.0)
    decayed, _ = train_baseline(package, epochs=20, seed=42, weight_decay=1.0)
    assert decayed.relation_weight.norm().item() < plain.relation_weight.norm().item()


def test_train_gnn_default_dims_are_64() -> None:
    package = load_graph_package(FIXTURE_DIR, seed=42)
    model, _ = train_gnn(package, epochs=1, seed=42)
    assert model.hidden_dim == 64
    assert model.embedding_dim == 64


def test_train_gnn_weight_decay_shrinks_relation_weight() -> None:
    package = load_graph_package(FIXTURE_DIR, seed=42)
    plain, _ = train_gnn(
        package, epochs=20, hidden_dim=8, embedding_dim=8, seed=42, weight_decay=0.0
    )
    decayed, _ = train_gnn(
        package, epochs=20, hidden_dim=8, embedding_dim=8, seed=42, weight_decay=1.0
    )
    assert decayed.relation_weight.norm().item() < plain.relation_weight.norm().item()


def test_train_gnn_reduces_loss_on_fixture_graph() -> None:
    package = load_graph_package(FIXTURE_DIR, seed=42)
    _, history = train_gnn(package, epochs=30, hidden_dim=8, embedding_dim=8, seed=42)
    # Each epoch's BCE batch is a fresh held-out supervision sample, so per-epoch
    # loss is noisy; compare windowed means rather than single endpoints.
    first, last = history.train_loss[:5], history.train_loss[-5:]
    assert sum(last) / len(last) < sum(first) / len(first)


def test_train_gnn_is_deterministic_for_fixed_seed() -> None:
    package = load_graph_package(FIXTURE_DIR, seed=42)
    _, history_a = train_gnn(package, epochs=5, hidden_dim=8, embedding_dim=8, seed=42)
    _, history_b = train_gnn(package, epochs=5, hidden_dim=8, embedding_dim=8, seed=42)
    assert history_a.train_loss == history_b.train_loss


def test_train_gnn_records_validation_loss_without_backprop() -> None:
    package = load_graph_package(FIXTURE_DIR, seed=42)
    _, history = train_gnn(package, epochs=5, hidden_dim=8, embedding_dim=8, seed=42)
    assert len(history.validation_loss) == 5
    assert len(history.train_loss) == 5


def test_train_gnn_without_patience_runs_full_epochs_and_tracks_best_epoch() -> None:
    package = load_graph_package(FIXTURE_DIR, seed=42)
    _, history = train_gnn(package, epochs=15, hidden_dim=8, embedding_dim=8, seed=42)
    assert len(history.train_loss) == 15
    assert 1 <= history.best_epoch <= 15
    assert history.validation_loss[history.best_epoch - 1] == min(history.validation_loss)


def test_train_gnn_with_patience_stops_early_and_restores_best_weights() -> None:
    package = load_graph_package(FIXTURE_DIR, seed=42)
    _, history = train_gnn(
        package, epochs=100, hidden_dim=8, embedding_dim=8, seed=42, patience=5
    )
    assert len(history.train_loss) < 100
    assert len(history.train_loss) == history.best_epoch + 5
    assert history.validation_loss[history.best_epoch - 1] == min(history.validation_loss)


# --- edge-level held-out supervision (GNN train-target leakage guard) --------


def _dg_pairs(data: HeteroData, package: GraphPackage) -> tuple[set, set]:
    """(forward, reverse) Disease-Gene pairs present in `data`, as id tuples."""
    disease_ids = package.node_indexes["Disease"].ids
    gene_ids = package.node_indexes["Gene"].ids
    forward = {
        (disease_ids[d], gene_ids[g])
        for d, g in data[DG_EDGE_TYPE].edge_index.t().tolist()
    }
    reverse = {
        (disease_ids[d], gene_ids[g])
        for g, d in data[DG_REVERSE_EDGE_TYPE].edge_index.t().tolist()
    }
    return forward, reverse


def test_split_message_supervision_is_seeded_partition() -> None:
    package = load_graph_package(FIXTURE_DIR, seed=42)
    train_pairs = package.positives["train"]
    message, supervision = split_message_supervision(
        train_pairs, supervision_ratio=0.5, generator=torch.Generator().manual_seed(42)
    )
    assert message == [("ORPHA:1", "HGNC:1"), ("ORPHA:2", "HGNC:2")]
    assert supervision == [("ORPHA:3", "HGNC:3"), ("ORPHA:4", "HGNC:4")]
    assert sorted(message + supervision) == sorted(train_pairs)


def test_split_message_supervision_rejects_bad_ratio() -> None:
    with pytest.raises(ValueError):
        split_message_supervision(
            [("D", "G")], supervision_ratio=0.0, generator=torch.Generator()
        )


def test_remove_dg_edges_drops_forward_and_reverse_without_mutating_input() -> None:
    package = load_graph_package(FIXTURE_DIR, seed=42)
    held_out = [("ORPHA:3", "HGNC:3")]
    pruned = remove_dg_edges(package.data, package.node_indexes, held_out)

    forward, reverse = _dg_pairs(pruned, package)
    assert ("ORPHA:3", "HGNC:3") not in forward
    assert ("ORPHA:3", "HGNC:3") not in reverse
    assert forward == reverse == set(package.positives["train"]) - set(held_out)
    # Non-DG edge types untouched; original package graph still complete.
    for edge_type in package.data.edge_types:
        if edge_type not in (DG_EDGE_TYPE, DG_REVERSE_EDGE_TYPE):
            assert torch.equal(pruned[edge_type].edge_index, package.data[edge_type].edge_index)
    full_forward, full_reverse = _dg_pairs(package.data, package)
    assert full_forward == full_reverse == set(package.positives["train"])


def test_epoch_batch_supervision_pairs_are_absent_from_encoded_graph() -> None:
    package = load_graph_package(FIXTURE_DIR, seed=42)
    generator = torch.Generator().manual_seed(42)

    # Fixed seed -> the first epoch holds out exactly ORPHA:3/HGNC:3.
    batch = build_gnn_epoch_batch(package, supervision_ratio=0.3, generator=generator)
    assert batch.supervision_pairs == [("ORPHA:3", "HGNC:3")]
    forward, reverse = _dg_pairs(batch.message_data, package)
    assert ("ORPHA:3", "HGNC:3") not in forward
    assert ("ORPHA:3", "HGNC:3") not in reverse
    assert forward == set(package.positives["train"]) - {("ORPHA:3", "HGNC:3")}

    # BCE positives are exactly the held-out pairs; negatives balance them 1:1.
    disease_ids = package.node_indexes["Disease"].ids
    gene_ids = package.node_indexes["Gene"].ids
    labelled = [
        ((disease_ids[d], gene_ids[g]), label)
        for d, g, label in zip(
            batch.disease_idx.tolist(), batch.gene_idx.tolist(), batch.labels.tolist()
        )
    ]
    positives = [pair for pair, label in labelled if label == 1.0]
    negatives = [pair for pair, label in labelled if label == 0.0]
    assert positives == batch.supervision_pairs
    assert negatives == batch.negative_pairs
    assert len(negatives) == len(positives)
    assert set(negatives) <= set(package.negatives["train"])

    # Re-sampled each epoch: a later draw holds out a different pair.
    second = build_gnn_epoch_batch(package, supervision_ratio=0.3, generator=generator)
    assert second.supervision_pairs == [("ORPHA:1", "HGNC:1")]


def test_train_gnn_never_encodes_a_supervised_pair_during_training(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = load_graph_package(FIXTURE_DIR, seed=42)
    train_pairs = set(package.positives["train"])
    records: list[tuple[bool, set, set]] = []
    decoded: list[set] = []
    original_encode = GNNLinkPredictor.encode
    original_decode = GNNLinkPredictor.decode

    def recording_encode(self: GNNLinkPredictor, data: HeteroData) -> dict:
        forward, reverse = _dg_pairs(data, package)
        records.append((self.training, forward, reverse))
        return original_encode(self, data)

    def recording_decode(self, disease_emb, gene_emb, disease_idx, gene_idx):  # type: ignore[no-untyped-def]
        if self.training:
            disease_ids = package.node_indexes["Disease"].ids
            gene_ids = package.node_indexes["Gene"].ids
            decoded.append(
                {(disease_ids[d], gene_ids[g]) for d, g in zip(disease_idx.tolist(), gene_idx.tolist())}
            )
        return original_decode(self, disease_emb, gene_emb, disease_idx, gene_idx)

    monkeypatch.setattr(GNNLinkPredictor, "encode", recording_encode)
    monkeypatch.setattr(GNNLinkPredictor, "decode", recording_decode)
    train_gnn(package, epochs=6, hidden_dim=8, embedding_dim=8, seed=42)

    training_graphs = [(fwd, rev) for is_training, fwd, rev in records if is_training]
    eval_graphs = [(fwd, rev) for is_training, fwd, rev in records if not is_training]
    assert len(training_graphs) == len(decoded) == 6
    held_out_seen: set = set()
    for (forward, reverse), scored in zip(training_graphs, decoded):
        supervised = scored & train_pairs
        assert supervised, "each epoch must score some train positives"
        # No train positive scored this epoch is in the graph it was encoded with.
        assert not (supervised & forward)
        assert not (supervised & reverse)
        # Every other train positive stays as message context.
        assert forward == reverse == train_pairs - supervised
        held_out_seen |= supervised
    # Different pairs are held out across epochs (fresh per-epoch split).
    assert len(held_out_seen) > 1
    # Validation encodes the full train message graph.
    assert eval_graphs and all(fwd == rev == train_pairs for fwd, rev in eval_graphs)
