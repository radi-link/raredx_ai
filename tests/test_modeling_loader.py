"""Tests for modeling CSV loading primitives."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import torch

from app.modeling.loader import (
    build_hetero_data,
    build_node_index,
    load_graph_package,
    load_message_edge_csv,
    load_negative_pairs_csv,
    load_node_csv,
    load_pair_csv,
    split_negative_pairs,
    validate_graph_package_invariants,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "modeling"


def test_load_node_csv_indexes_by_node_id_and_parses_rankable() -> None:
    nodes = load_node_csv(FIXTURE_DIR / "gene_nodes.csv")
    assert nodes["HGNC:1"]["node_type"] == "Gene"
    assert nodes["HGNC:5"]["rankable"] is False


def test_load_message_edge_csv_returns_all_relations() -> None:
    edges = load_message_edge_csv(FIXTURE_DIR / "training_message_edges.csv")
    relations = {e["relation"] for e in edges}
    assert relations == {"IS_A", "HAS_PHENOTYPE", "ASSOCIATED_WITH"}


def test_load_pair_csv_rejects_non_associated_with_rows(tmp_path: Path) -> None:
    bad = tmp_path / "bad.csv"
    bad.write_text("source_id,target_id,relation\nORPHA:1,HP:0000001,HAS_PHENOTYPE\n")
    with pytest.raises(ValueError):
        load_pair_csv(bad)


def test_load_negative_pairs_csv_returns_label_zero_pairs() -> None:
    negatives = load_negative_pairs_csv(FIXTURE_DIR / "negative_pairs.csv")
    assert len(negatives) == 6
    assert all(isinstance(p, tuple) and len(p) == 2 for p in negatives)


def test_split_negative_pairs_matches_requested_sizes_and_is_disjoint() -> None:
    negatives = load_negative_pairs_csv(FIXTURE_DIR / "negative_pairs.csv")
    result = split_negative_pairs(negatives, split_sizes={"train": 4, "validation": 1, "test": 1})
    assert [len(result[name]) for name in ("train", "validation", "test")] == [4, 1, 1]
    seen = [p for group in result.values() for p in group]
    assert len(seen) == len(set(seen))


def test_split_negative_pairs_is_deterministic_for_fixed_seed() -> None:
    negatives = load_negative_pairs_csv(FIXTURE_DIR / "negative_pairs.csv")
    first = split_negative_pairs(negatives, split_sizes={"train": 4, "validation": 1, "test": 1}, seed=42)
    second = split_negative_pairs(list(reversed(negatives)), split_sizes={"train": 4, "validation": 1, "test": 1}, seed=42)
    assert first == second


def test_split_negative_pairs_rejects_oversized_request() -> None:
    negatives = load_negative_pairs_csv(FIXTURE_DIR / "negative_pairs.csv")
    with pytest.raises(ValueError):
        split_negative_pairs(negatives, split_sizes={"train": 10, "validation": 1, "test": 1})


def test_build_node_index_separates_rankable_ids() -> None:
    nodes = load_node_csv(FIXTURE_DIR / "gene_nodes.csv")
    index = build_node_index(nodes)
    assert "HGNC:5" in index.ids and "HGNC:5" not in index.rankable
    assert "HGNC:1" in index.rankable
    assert index.index["HGNC:1"] == index.ids.index("HGNC:1")


def test_build_hetero_data_has_forward_and_reverse_edge_types() -> None:
    genes = build_node_index(load_node_csv(FIXTURE_DIR / "gene_nodes.csv"))
    diseases = build_node_index(load_node_csv(FIXTURE_DIR / "disease_nodes.csv"))
    phenotypes = build_node_index(load_node_csv(FIXTURE_DIR / "hpo_nodes.csv"))
    edges = load_message_edge_csv(FIXTURE_DIR / "training_message_edges.csv")
    data = build_hetero_data(
        {"Gene": genes, "Disease": diseases, "Phenotype": phenotypes}, edges
    )
    assert ("Disease", "ASSOCIATED_WITH", "Gene") in data.edge_types
    assert ("Gene", "rev_ASSOCIATED_WITH", "Disease") in data.edge_types
    assert ("Phenotype", "rev_IS_A", "Phenotype") in data.edge_types
    assert ("Phenotype", "rev_HAS_PHENOTYPE", "Disease") in data.edge_types
    forward = data["Disease", "ASSOCIATED_WITH", "Gene"].edge_index
    reverse = data["Gene", "rev_ASSOCIATED_WITH", "Disease"].edge_index
    assert forward.shape[1] == reverse.shape[1] == 4
    assert torch.equal(forward[[0, 1]], reverse[[1, 0]])


def _copy_fixture(tmp_path: Path) -> Path:
    """Copy the clean fixture package into `tmp_path` so one file can be corrupted."""
    target = tmp_path / "graph"
    shutil.copytree(FIXTURE_DIR, target)
    return target


def _append_line(path: Path, line: str) -> None:
    text = path.read_text(encoding="utf-8")
    if not text.endswith("\n"):
        text += "\n"
    path.write_text(text + line + "\n", encoding="utf-8")


def test_load_graph_package_message_graph_excludes_val_test_pairs() -> None:
    # Fixture's cold-start test pair is ORPHA:1/HGNC:5; validation pair is ORPHA:2/HGNC:3.
    package = load_graph_package(FIXTURE_DIR, seed=42)
    disease_ids = package.node_indexes["Disease"].ids
    gene_ids = package.node_indexes["Gene"].ids
    edge_index = package.data["Disease", "ASSOCIATED_WITH", "Gene"].edge_index
    message_pairs = {
        (disease_ids[d], gene_ids[g]) for d, g in edge_index.t().tolist()
    }
    heldout = set(package.positives["validation"]) | set(package.positives["test"])
    assert ("ORPHA:1", "HGNC:5") in heldout
    assert not (message_pairs & heldout)
    assert message_pairs == set(package.positives["train"])


def test_load_graph_package_rejects_validation_pair_in_message_graph(tmp_path: Path) -> None:
    graph_dir = _copy_fixture(tmp_path)
    _append_line(
        graph_dir / "training_message_edges.csv",
        "ORPHA:2,HGNC:3,ASSOCIATED_WITH" + "," * 29,
    )
    with pytest.raises(ValueError, match="leakage"):
        load_graph_package(graph_dir, seed=42)


def test_load_graph_package_rejects_test_pair_in_message_graph(tmp_path: Path) -> None:
    graph_dir = _copy_fixture(tmp_path)
    _append_line(
        graph_dir / "training_message_edges.csv",
        "ORPHA:1,HGNC:5,ASSOCIATED_WITH" + "," * 29,
    )
    with pytest.raises(ValueError, match="leakage"):
        load_graph_package(graph_dir, seed=42)


def test_load_graph_package_rejects_train_pair_missing_from_message_graph(
    tmp_path: Path,
) -> None:
    graph_dir = _copy_fixture(tmp_path)
    edges_path = graph_dir / "training_message_edges.csv"
    lines = edges_path.read_text(encoding="utf-8").splitlines()
    kept = [line for line in lines if not line.startswith("ORPHA:4,HGNC:4,")]
    assert len(kept) == len(lines) - 1
    edges_path.write_text("\n".join(kept) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must equal the train positives"):
        load_graph_package(graph_dir, seed=42)


def test_load_graph_package_rejects_overlapping_positive_splits(tmp_path: Path) -> None:
    graph_dir = _copy_fixture(tmp_path)
    _append_line(graph_dir / "splits" / "test.csv", "ORPHA:1,HGNC:1,ASSOCIATED_WITH")
    with pytest.raises(ValueError, match="positive splits 'train' and 'test' overlap"):
        load_graph_package(graph_dir, seed=42)


def test_load_graph_package_rejects_negative_that_is_a_known_positive(
    tmp_path: Path,
) -> None:
    graph_dir = _copy_fixture(tmp_path)
    negatives_path = graph_dir / "negative_pairs.csv"
    # Swap one negative for the validation positive; pool stays at 6 pairs, so
    # every negative is assigned to some split.
    text = negatives_path.read_text(encoding="utf-8")
    negatives_path.write_text(
        text.replace("ORPHA:1,HGNC:2,0", "ORPHA:2,HGNC:3,0"), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="known positive"):
        load_graph_package(graph_dir, seed=42)


def test_load_graph_package_rejects_unknown_node_id(tmp_path: Path) -> None:
    graph_dir = _copy_fixture(tmp_path)
    test_path = graph_dir / "splits" / "test.csv"
    text = test_path.read_text(encoding="utf-8")
    test_path.write_text(text.replace("ORPHA:1,HGNC:5,", "ORPHA:99,HGNC:5,"), encoding="utf-8")
    with pytest.raises(ValueError, match="not a Disease node"):
        load_graph_package(graph_dir, seed=42)


def test_validate_invariants_rejects_overlapping_negative_splits() -> None:
    package = load_graph_package(FIXTURE_DIR, seed=42)
    shared = package.negatives["train"][0]
    negatives = {**package.negatives, "test": [shared]}
    message_edges = [
        {"source_id": d, "target_id": g, "relation": "ASSOCIATED_WITH"}
        for d, g in package.positives["train"]
    ]
    with pytest.raises(ValueError, match="negative splits 'train' and 'test' overlap"):
        validate_graph_package_invariants(
            package.node_indexes, message_edges, package.positives, negatives
        )


def test_load_graph_package_end_to_end_from_fixtures() -> None:
    package = load_graph_package(FIXTURE_DIR, seed=42)
    assert set(package.positives) == {"train", "validation", "test"}
    assert len(package.negatives["train"]) == len(package.positives["train"])
    assert ("Disease", "ASSOCIATED_WITH", "Gene") in package.data.edge_types
    train_pairs = set(package.positives["train"])
    val_test_pairs = set(package.positives["validation"]) | set(package.positives["test"])
    assert not (train_pairs & val_test_pairs)
