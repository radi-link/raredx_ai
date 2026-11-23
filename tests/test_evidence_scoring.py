from __future__ import annotations

import math

import pytest

from app.etl.evidence_scoring import (
    aggregate_disease_gene,
    apply_gene_idf,
    fit_gene_idf,
    score_associations,
)

TYPE_WEIGHTS = [
    ("Disease-causing germline mutation(s) in", 1.0),
    ("Disease-causing germline mutation(s) (loss of function) in", 1.0),
    ("Disease-causing germline mutation(s) (gain of function) in", 1.0),
    ("Disease-causing somatic mutation(s) in", 0.8),
    ("Major susceptibility factor in", 0.6),
    ("Modifying germline mutation in", 0.5),
    ("Part of a fusion gene in", 0.5),
    ("Role in the phenotype of", 0.4),
    ("Candidate gene tested in", 0.3),
    ("Biomarker tested in", 0.2),
]


@pytest.mark.parametrize(("association_type", "expected_weight"), TYPE_WEIGHTS)
def test_score_associations_reproduces_v1_type_weights(
    association_type: str,
    expected_weight: float,
) -> None:
    [scored] = score_associations(
        [
            {
                "disease_id": "ORPHA:1",
                "gene_id": "HGNC:1",
                "association_type": association_type,
                "association_status": "Assessed",
            }
        ]
    )

    assert scored["association_type_weight"] == expected_weight
    assert scored["association_status_weight"] == 1.0
    assert scored["evidence_score"] == expected_weight
    assert scored["weight_rule_version"] == "v1.0"


def test_score_associations_defaults_missing_status_without_mutating_input() -> None:
    raw = {
        "disease_id": "ORPHA:1",
        "gene_id": "HGNC:1",
        "association_type": "Disease-causing germline mutation(s) in",
        "association_status": None,
        "provenance": {"row": 7},
    }

    [scored] = score_associations([raw])

    assert raw["association_status"] is None
    assert scored["association_status"] == "missing"
    assert scored["association_status_weight"] == 0.5
    assert scored["evidence_score"] == 0.5
    assert scored["provenance"] == {"row": 7}


@pytest.mark.parametrize(
    "field,value",
    [
        ("association_type", "Unknown association"),
        ("association_status", "Unknown status"),
    ],
)
def test_score_associations_rejects_unknown_evidence_values(
    field: str, value: str
) -> None:
    row = {
        "disease_id": "ORPHA:1",
        "gene_id": "HGNC:1",
        "association_type": "Disease-causing germline mutation(s) in",
        "association_status": "Assessed",
    }
    row[field] = value

    with pytest.raises(ValueError, match=field):
        score_associations([row])


def test_aggregate_disease_gene_uses_max_and_preserves_all_evidence() -> None:
    scored = score_associations(
        [
            {
                "disease_id": "ORPHA:1",
                "gene_id": "HGNC:1",
                "association_type": "Major susceptibility factor in",
                "association_status": "Assessed",
                "source_validation": "PMID:2",
                "source_database": "Orphanet",
                "provenance": {"row": 2},
            },
            {
                "disease_id": "ORPHA:1",
                "gene_id": "HGNC:1",
                "association_type": "Disease-causing germline mutation(s) in",
                "association_status": "Not assessed",
                "source_validation": "PMID:1",
                "source_database": "Orphanet",
                "provenance": ["xml", 1],
            },
            {
                "disease_id": "ORPHA:1",
                "gene_id": "HGNC:1",
                "association_type": "Major susceptibility factor in",
                "association_status": "Assessed",
                "source_validation": "PMID:2",
                "source_database": "Orphanet",
                "provenance": {"row": 2},
            },
        ]
    )

    [edge] = aggregate_disease_gene(scored)

    assert edge["source_id"] == "ORPHA:1"
    assert edge["target_id"] == "HGNC:1"
    assert edge["relation"] == "ASSOCIATED_WITH"
    assert edge["evidence_score"] == 0.6
    assert edge["evidence_count"] == 3
    assert edge["association_type"] == [
        "Disease-causing germline mutation(s) in",
        "Major susceptibility factor in",
    ]
    assert edge["association_status"] == ["Assessed", "Not assessed"]
    assert edge["source_validation"] == ["PMID:1", "PMID:2"]
    assert edge["source_database"] == ["Orphanet"]
    assert len(edge["evidence_provenance"]) == 3
    assert edge["evidence_provenance"].count(edge["evidence_provenance"][1]) == 2


def test_aggregate_disease_gene_is_deterministic_for_reordered_input() -> None:
    scored = score_associations(
        [
            {
                "disease_id": "ORPHA:2",
                "gene_id": "HGNC:2",
                "association_type": "Candidate gene tested in",
                "association_status": "missing",
                "source_validation": "B",
                "source_database": "Orphanet",
            },
            {
                "disease_id": "ORPHA:1",
                "gene_id": "HGNC:1",
                "association_type": "Disease-causing somatic mutation(s) in",
                "association_status": "Assessed",
                "source_validation": "A",
                "source_database": "Orphanet",
            },
        ]
    )

    assert aggregate_disease_gene(scored) == aggregate_disease_gene(
        list(reversed(scored))
    )


def test_fit_and_apply_gene_idf_use_hand_derived_values() -> None:
    edges = [
        {"source_id": "ORPHA:1", "target_id": "HGNC:1", "evidence_score": 0.8},
        {"source_id": "ORPHA:2", "target_id": "HGNC:1", "evidence_score": 0.7},
        {"source_id": "ORPHA:2", "target_id": "HGNC:2", "evidence_score": 0.6},
    ]

    fit = fit_gene_idf(edges, scope="train_only")
    applied = apply_gene_idf(edges, fit)

    assert fit["disease_count_total"] == 2
    assert fit["gene_df"] == {"HGNC:1": 2, "HGNC:2": 1}
    assert fit["pairs"] == [
        ["ORPHA:1", "HGNC:1"],
        ["ORPHA:2", "HGNC:1"],
        ["ORPHA:2", "HGNC:2"],
    ]
    assert (
        fit["fit_pair_sha256"]
        == "00dd47d4004d02c5979474dd238c4a552f0ace0870296b582cdabe8c41edb2ad"
    )
    assert fit["idf_scope"] == "train_only"
    assert fit["idf_min"] == 0.0
    assert fit["idf_max"] == pytest.approx(math.log(1.5))

    by_pair = {(edge["source_id"], edge["target_id"]): edge for edge in applied}
    common = by_pair[("ORPHA:1", "HGNC:1")]
    rare = by_pair[("ORPHA:2", "HGNC:2")]
    assert common["gene_df"] == 2
    assert common["idf_raw"] == 0.0
    assert common["idf_norm"] == 0.0
    assert common["idf_penalty_weight"] == 0.5
    assert common["evidence_score"] == 0.8
    assert common["evidence_score_idf"] == 0.4
    assert rare["gene_df"] == 1
    assert rare["idf_norm"] == 1.0
    assert rare["idf_penalty_weight"] == 1.0
    assert rare["evidence_score_idf"] == 0.6
    assert rare["idf_scope"] == "train_only"
    assert rare["idf_rule_version"] == "idf_v1.0"
    assert rare["disease_count_total"] == 2


def test_fit_gene_idf_deduplicates_pairs_and_is_input_order_independent() -> None:
    edges = [
        {"source_id": "ORPHA:2", "target_id": "HGNC:1", "evidence_score": 0.7},
        {"source_id": "ORPHA:1", "target_id": "HGNC:1", "evidence_score": 0.8},
        {"source_id": "ORPHA:1", "target_id": "HGNC:1", "evidence_score": 0.4},
    ]

    fit = fit_gene_idf(edges, scope="full_reference")

    assert fit == fit_gene_idf(list(reversed(edges)), scope="full_reference")
    assert fit["pairs"] == [["ORPHA:1", "HGNC:1"], ["ORPHA:2", "HGNC:1"]]
    assert fit["gene_df"] == {"HGNC:1": 2}


def test_degenerate_and_empty_idf_fits_are_finite_and_auditable() -> None:
    one_gene = [{"source_id": "ORPHA:1", "target_id": "HGNC:1", "evidence_score": 0.8}]
    degenerate_fit = fit_gene_idf(one_gene, scope="full_reference")
    [degenerate] = apply_gene_idf(one_gene, degenerate_fit)

    assert degenerate["idf_norm"] == 1.0
    assert degenerate["idf_penalty_weight"] == 1.0
    assert degenerate["evidence_score_idf"] == 0.8

    empty_fit = fit_gene_idf([], scope="train_only")
    [unseen] = apply_gene_idf(
        [{"source_id": "ORPHA:9", "target_id": "HGNC:9", "evidence_score": 0.4}],
        empty_fit,
    )

    assert empty_fit["disease_count_total"] == 0
    assert empty_fit["pairs"] == []
    assert empty_fit["gene_df"] == {}
    assert empty_fit["idf_min"] == 0.0
    assert empty_fit["idf_max"] == 0.0
    assert unseen["gene_df"] == 0
    assert unseen["idf_raw"] == 0.0
    assert unseen["idf_norm"] == 1.0
    assert unseen["idf_penalty_weight"] == 1.0
    assert math.isfinite(unseen["evidence_score_idf"])


def test_heldout_edges_use_frozen_fit_without_mutating_it() -> None:
    train = [
        {"source_id": "ORPHA:1", "target_id": "HGNC:1", "evidence_score": 1.0},
        {"source_id": "ORPHA:2", "target_id": "HGNC:1", "evidence_score": 1.0},
        {"source_id": "ORPHA:2", "target_id": "HGNC:2", "evidence_score": 1.0},
    ]
    fit = fit_gene_idf(train, scope="train_only")
    frozen_snapshot = dict(fit)

    [unseen] = apply_gene_idf(
        [{"source_id": "ORPHA:99", "target_id": "HGNC:99", "evidence_score": 0.5}],
        fit,
    )

    assert fit == frozen_snapshot
    assert unseen["gene_df"] == 0
    assert unseen["idf_raw"] == pytest.approx(math.log(3.0))
    assert unseen["idf_norm"] == 1.0
    assert unseen["idf_penalty_weight"] == 1.0
    assert unseen["evidence_score_idf"] == 0.5


@pytest.mark.parametrize("scope", ["", "test_only", "FULL_REFERENCE"])
def test_fit_gene_idf_rejects_unknown_scope(scope: str) -> None:
    with pytest.raises(ValueError, match="scope"):
        fit_gene_idf([], scope=scope)
