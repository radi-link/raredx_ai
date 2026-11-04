import math
import random

import pytest

from app.graph.splits import (
    sample_negative_pairs,
    split_disease_gene,
    training_message_edges,
    validate_split_leakage,
    with_reverse_edges,
)


def _edge(disease: str, gene: str, **extra: object) -> dict[str, object]:
    return {
        "source_id": disease,
        "target_id": gene,
        "relation": "ASSOCIATED_WITH",
        **extra,
    }


def test_split_disease_gene_is_order_independent_and_uses_local_rng() -> None:
    edges = [_edge(f"ORPHA:{index + 1}", f"HGNC:{index + 1}") for index in range(10)]
    state_before = random.getstate()

    first = split_disease_gene(edges, seed=17, val_fraction=0.2, test_fraction=0.2)
    second = split_disease_gene(
        list(reversed(edges)), seed=17, val_fraction=0.2, test_fraction=0.2
    )

    assert first == second
    assert {name: len(rows) for name, rows in first.items()} == {
        "train": 6,
        "validation": 2,
        "test": 2,
    }
    assert random.getstate() == state_before
    assert all(
        row["split"] == split_name
        and row["split_version"] == "dg_link_v1"
        and row["seed"] == 17
        for split_name, rows in first.items()
        for row in rows
    )


@pytest.mark.parametrize(
    ("val_fraction", "test_fraction"),
    [
        (-0.1, 0.1),
        (0.1, -0.1),
        (0.5, 0.5),
        (0.6, 0.5),
        (math.nan, 0.1),
        (0.1, math.inf),
    ],
)
def test_split_disease_gene_rejects_invalid_ratios(
    val_fraction: float, test_fraction: float
) -> None:
    with pytest.raises(ValueError):
        split_disease_gene(
            [_edge("ORPHA:1", "HGNC:1")],
            val_fraction=val_fraction,
            test_fraction=test_fraction,
        )


def test_split_disease_gene_handles_empty_edges_and_zero_holdout() -> None:
    assert split_disease_gene([], val_fraction=0.0, test_fraction=0.0) == {
        "train": [],
        "validation": [],
        "test": [],
    }

    result = split_disease_gene(
        [_edge("ORPHA:1", "HGNC:1")], val_fraction=0.0, test_fraction=0.0
    )
    assert [row["split"] for row in result["train"]] == ["train"]
    assert result["validation"] == []
    assert result["test"] == []


def test_split_disease_gene_rejects_duplicate_or_unsupported_targets() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        split_disease_gene(
            [
                _edge("ORPHA:1", "HGNC:1"),
                _edge("ORPHA:1", "HGNC:1", evidence_score=0.5),
            ]
        )

    with pytest.raises(ValueError, match="ASSOCIATED_WITH"):
        split_disease_gene(
            [
                {
                    "source_id": "ORPHA:1",
                    "target_id": "HP:0000001",
                    "relation": "HAS_PHENOTYPE",
                }
            ]
        )

    with pytest.raises(ValueError, match="forward"):
        split_disease_gene(
            [
                {
                    "source_id": "HGNC:1",
                    "target_id": "ORPHA:1",
                    "relation": "ASSOCIATED_WITH_REVERSE",
                }
            ]
        )


@pytest.mark.parametrize(
    "edge",
    [
        _edge("ORPHA:1", "HGNC:1", rankable=False),
        _edge("ORPHA:1", "ORPHANET_LOCUS:DHS6S1"),
        _edge("ORPHA:1", "HGNC:1", unresolved=True),
    ],
)
def test_split_disease_gene_rejects_non_rankable_targets(
    edge: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="rankable"):
        split_disease_gene([edge])


def test_with_reverse_edges_preserves_original_and_links_reverse() -> None:
    original = _edge(
        "ORPHA:1",
        "HGNC:1",
        edge_id="dg-1",
        split="train",
        split_version="dg_link_v1",
        seed=42,
    )

    expanded = with_reverse_edges([original])

    assert expanded[0] == original
    assert expanded[0] is not original
    assert original["relation"] == "ASSOCIATED_WITH"
    assert expanded[1] == {
        **original,
        "source_id": "HGNC:1",
        "target_id": "ORPHA:1",
        "relation": "ASSOCIATED_WITH_REVERSE",
        "edge_id": "dg-1:reverse",
        "reverse_of": "dg-1",
    }


def test_with_reverse_edges_uses_deterministic_pair_key_without_edge_id() -> None:
    expanded = with_reverse_edges([_edge("ORPHA:2", "HGNC:7", split="validation")])

    assert expanded[1]["reverse_of"] == "ORPHA:2|ASSOCIATED_WITH|HGNC:7"
    assert expanded[1]["edge_id"] == ("ORPHA:2|ASSOCIATED_WITH|HGNC:7:reverse")
    assert expanded[1]["split"] == "validation"


def test_sample_negative_pairs_is_deterministic_unique_and_excludes_positives() -> None:
    diseases = ["ORPHA:3", "ORPHA:1", "ORPHA:2"]
    genes = ["HGNC:2", "HGNC:1"]
    positives = {("ORPHA:1", "HGNC:1"), ("ORPHA:2", "HGNC:2")}

    first = sample_negative_pairs(diseases, genes, positives, count=4, seed=9)
    second = sample_negative_pairs(
        list(reversed(diseases)),
        list(reversed(genes)),
        positives,
        count=4,
        seed=9,
    )

    assert first == second
    assert len(first) == 4
    assert len(set(first)) == 4
    assert not set(first) & positives


def test_sample_negative_pairs_proves_capacity_and_never_returns_short() -> None:
    positives = {("ORPHA:1", "HGNC:1"), ("ORPHA:1", "HGNC:2")}

    assert (
        sample_negative_pairs(["ORPHA:1"], ["HGNC:1", "HGNC:2"], positives, count=0)
        == []
    )
    with pytest.raises(ValueError, match="capacity"):
        sample_negative_pairs(["ORPHA:1"], ["HGNC:1", "HGNC:2"], positives, count=1)
    with pytest.raises(ValueError, match="count"):
        sample_negative_pairs(["ORPHA:1"], ["HGNC:1"], set(), count=-1)


def test_sample_negative_pairs_handles_large_cartesian_space_without_materializing_it() -> (
    None
):
    diseases = [f"ORPHA:{index + 1}" for index in range(100_000)]
    genes = [f"HGNC:{index + 1}" for index in range(100_000)]

    sampled = sample_negative_pairs(
        diseases,
        genes,
        {("ORPHA:1", "HGNC:1")},
        count=3,
        seed=3,
    )

    assert len(sampled) == 3
    assert ("ORPHA:1", "HGNC:1") not in sampled


def test_validate_split_leakage_detects_pair_overlap_including_reverse() -> None:
    splits = {
        "train": [_edge("ORPHA:1", "HGNC:1", split="train")],
        "validation": [
            {
                "source_id": "HGNC:1",
                "target_id": "ORPHA:1",
                "relation": "ASSOCIATED_WITH_REVERSE",
                "split": "validation",
                "reverse_of": "dg-1",
            }
        ],
        "test": [],
    }

    report = validate_split_leakage(splits)

    assert report["passed"] is False
    assert report["counts"]["pair_overlap_across_splits"] == 1
    assert report["counts"]["reverse_split_conflicts"] == 1


def test_validate_split_leakage_detects_duplicates_and_positive_negatives() -> None:
    duplicate = _edge("ORPHA:1", "HGNC:1", split="train")
    splits = {
        "train": [duplicate, dict(duplicate)],
        "validation": [],
        "test": [],
    }

    report = validate_split_leakage(
        splits,
        known_positive_pairs={("ORPHA:1", "HGNC:1")},
        negative_pairs=[("ORPHA:1", "HGNC:1"), ("ORPHA:2", "HGNC:2")],
    )

    assert report["passed"] is False
    assert report["counts"]["duplicate_pairs_within_split"] == 1
    assert report["counts"]["known_positive_negative_collisions"] == 1


def test_validate_split_leakage_checks_train_only_idf_fit_contract() -> None:
    splits = {
        "train": [
            _edge("ORPHA:1", "HGNC:1", split="train"),
            _edge("ORPHA:2", "HGNC:1", split="train"),
            _edge("ORPHA:3", "HGNC:2", split="train"),
        ],
        "validation": [_edge("ORPHA:4", "HGNC:3", split="validation")],
        "test": [],
    }
    from app.etl.evidence_scoring import fit_gene_idf

    valid_fit = fit_gene_idf(splits["train"], scope="train_only")

    valid_report = validate_split_leakage(splits, idf_fit=valid_fit)
    invalid_report = validate_split_leakage(
        splits,
        idf_fit={
            **valid_fit,
            "idf_scope": "full_reference",
            "disease_count_total": 4,
        },
    )

    assert valid_report["passed"] is True
    assert valid_report["counts"]["idf_fit_metadata_errors"] == 0
    assert invalid_report["passed"] is False
    assert invalid_report["counts"]["idf_fit_metadata_errors"] == 2


def test_training_message_edges_keeps_structural_and_train_dg_only() -> None:
    structural = [
        {"source_id": "HP:2", "target_id": "HP:1", "relation": "IS_A"},
        {
            "source_id": "ORPHA:1",
            "target_id": "HP:2",
            "relation": "HAS_PHENOTYPE",
        },
        {
            "source_id": "HGNC:1",
            "target_id": "HP:2",
            "relation": "ASSOCIATED_PHENOTYPE",
            "disease_id": "ORPHA:1",
        },
        {
            "source_id": "HGNC:2",
            "target_id": "HP:3",
            "relation": "ASSOCIATED_PHENOTYPE",
            "disease_id": "ORPHA:2",
        },
    ]
    splits = {
        "train": [_edge("ORPHA:1", "HGNC:1", split="train")],
        "validation": [_edge("ORPHA:2", "HGNC:2", split="validation")],
        "test": [_edge("ORPHA:3", "HGNC:3", split="test")],
    }

    message_edges = training_message_edges(structural, splits)

    assert [row["relation"] for row in message_edges] == [
        "IS_A",
        "HAS_PHENOTYPE",
        "ASSOCIATED_WITH",
    ]
    assert all(row.get("disease_id") != "ORPHA:2" for row in message_edges)
    assert not any(
        row["source_id"] == "ORPHA:2" and row["target_id"] == "HGNC:2"
        for row in message_edges
    )


def test_training_message_edges_rejects_prediction_edges_in_structural_input() -> None:
    with pytest.raises(ValueError, match="structural"):
        training_message_edges(
            [_edge("ORPHA:1", "HGNC:1")],
            {"train": [], "validation": [], "test": []},
        )


@pytest.mark.parametrize(
    "edge",
    [
        _edge("OMIM:1", "HGNC:1"),
        _edge("ORPHA:x", "HGNC:1"),
        _edge("ORPHA:1", "HGNC:x"),
        _edge("ORPHA:1", "HGNC:1", rankable=0),
    ],
)
def test_split_rejects_invalid_rankable_identifiers(edge: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="rankable"):
        split_disease_gene([edge])


def test_training_rejects_phenotype_prediction_context() -> None:
    with pytest.raises(ValueError, match="prediction_target"):
        training_message_edges(
            [],
            {"train": [], "validation": [], "test": []},
            prediction_target="disease_phenotype",
        )


@pytest.mark.parametrize(
    "field,value", [("reverse_of", "wrong"), ("seed", 99), ("source_id", "ORPHA:1")]
)
def test_validator_rejects_malformed_reverse(field: str, value: object) -> None:
    rows = with_reverse_edges([_edge("ORPHA:1", "HGNC:1", split="train", seed=42)])
    rows[1][field] = value
    assert (
        validate_split_leakage({"train": rows, "validation": [], "test": []})["passed"]
        is False
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("pairs", []),
        ("gene_df", {}),
        ("idf_min", 1.0),
        ("idf_max", 1.0),
        ("fit_pair_sha256", "bad"),
    ],
)
def test_validator_checks_exact_idf_fit(field: str, value: object) -> None:
    from app.etl.evidence_scoring import fit_gene_idf

    rows = [_edge("ORPHA:1", "HGNC:1", split="train")]
    fit = fit_gene_idf(rows, scope="train_only")
    fit[field] = value
    report = validate_split_leakage(
        {"train": rows, "validation": [], "test": []}, idf_fit=fit
    )
    assert report["counts"]["idf_fit_metadata_errors"] == 1


def test_negative_sampling_rejects_unresolved_identifiers() -> None:
    with pytest.raises(ValueError, match="rankable"):
        sample_negative_pairs(["ORPHA:1"], ["ORPHANET_LOCUS:abc"], set(), count=1)


@pytest.mark.parametrize(
    "disease,gene",
    [
        ("ORPHA:0", "HGNC:1"),
        ("ORPHA:01", "HGNC:1"),
        ("ORPHA:1", "HGNC:0"),
        ("ORPHA:1", "HGNC:01"),
    ],
)
def test_targets_require_canonical_positive_numeric_ids(
    disease: str, gene: str
) -> None:
    with pytest.raises(ValueError, match="rankable"):
        split_disease_gene([_edge(disease, gene)])
    with pytest.raises(ValueError, match="rankable"):
        sample_negative_pairs([disease], [gene], set(), count=1)


@pytest.mark.parametrize("name,label", [("other", "other"), ("train", "test")])
def test_validator_rejects_unknown_split_or_wrong_row_label(
    name: str, label: str
) -> None:
    report = validate_split_leakage({name: [_edge("ORPHA:1", "HGNC:1", split=label)]})
    assert report["passed"] is False
