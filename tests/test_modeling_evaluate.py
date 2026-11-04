"""Tests for the filtered bidirectional link-prediction evaluator."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.modeling.evaluate import (
    CandidateSubsetMetrics,
    EvaluationReport,
    MultiDiseaseCandidateMetrics,
    RankingMetrics,
    evaluate_candidate_subset_ranking,
    evaluate_link_prediction,
    evaluate_multi_disease_candidate_pool,
)
from app.modeling.loader import (
    GraphPackage,
    build_hetero_data,
    build_node_index,
    load_graph_package,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "modeling"

Pair = tuple[str, str]


class RecordingScorer:
    """Constant scorer that records every batch of pairs it is asked to score."""

    def __init__(self, value: float = 0.0) -> None:
        self.value = value
        self.batches: list[list[Pair]] = []

    def __call__(self, pairs: list[Pair]) -> list[float]:
        self.batches.append(list(pairs))
        return [self.value for _ in pairs]


def _batches_for(scorer: RecordingScorer, true_pair: Pair) -> tuple[list[Pair], list[Pair]]:
    """Return (gene_corruption_batch, disease_corruption_batch) recorded for `true_pair`."""
    disease_id, gene_id = true_pair
    gene_batches = [
        batch
        for batch in scorer.batches
        if true_pair in batch and all(d == disease_id for d, _ in batch)
        and any(g != gene_id for _, g in batch)
    ]
    disease_batches = [
        batch
        for batch in scorer.batches
        if true_pair in batch and all(g == gene_id for _, g in batch)
        and any(d != disease_id for d, _ in batch)
    ]
    assert len(gene_batches) == 1
    assert len(disease_batches) == 1
    return gene_batches[0], disease_batches[0]


# --- brief-specified tests ---------------------------------------------------


def test_evaluate_link_prediction_ranks_known_pair_first() -> None:
    package = load_graph_package(FIXTURE_DIR, seed=42)

    def perfect_scorer(pairs: list[Pair]) -> list[float]:
        true_pairs = set(package.positives["validation"])
        return [10.0 if pair in true_pairs else -10.0 for pair in pairs]

    report = evaluate_link_prediction(perfect_scorer, package, "validation")
    assert isinstance(report, EvaluationReport)
    assert report.combined.count == 2
    assert report.combined.hits_at_1 == 1.0
    assert report.combined.mrr == 1.0


def test_evaluate_link_prediction_separates_cold_start_pairs() -> None:
    package = load_graph_package(FIXTURE_DIR, seed=42)

    def constant_scorer(pairs: list[Pair]) -> list[float]:
        return [0.0 for _ in pairs]

    report = evaluate_link_prediction(constant_scorer, package, "test")
    # fixture's ORPHA:1/HGNC:5 test pair has no train-graph edge for HGNC:5
    assert report.cold_start_gene_corruption.count >= 1
    assert (
        report.cold_start_gene_corruption.count + report.gene_corruption.count
        == len(package.positives["test"])
    )


def test_evaluate_link_prediction_excludes_non_rankable_candidates() -> None:
    package = load_graph_package(FIXTURE_DIR, seed=42)
    rankable_genes = package.node_indexes["Gene"].rankable
    assert "HGNC:5" not in rankable_genes

    scorer = RecordingScorer()
    evaluate_link_prediction(scorer, package, "validation")
    scored_genes = {gene for batch in scorer.batches for _, gene in batch}
    # HGNC:5 is non-rankable and is not the true gene of any validation pair,
    # so it must never be offered as a corruption candidate.
    assert "HGNC:5" not in scored_genes


# --- candidate pool construction ---------------------------------------------


def test_candidate_pools_are_filtered_and_restricted_to_rankable() -> None:
    package = load_graph_package(FIXTURE_DIR, seed=42)
    scorer = RecordingScorer()
    evaluate_link_prediction(scorer, package, "validation")

    true_pair = ("ORPHA:2", "HGNC:3")
    gene_batch, disease_batch = _batches_for(scorer, true_pair)

    # Gene side: rankable {1,2,3,4}; HGNC:2 removed (train positive with ORPHA:2);
    # HGNC:5 never present (non-rankable). True pair appears exactly once.
    assert sorted(gene_batch) == [
        ("ORPHA:2", "HGNC:1"),
        ("ORPHA:2", "HGNC:3"),
        ("ORPHA:2", "HGNC:4"),
    ]
    # Disease side: ORPHA:3 removed (train positive with HGNC:3).
    assert sorted(disease_batch) == [
        ("ORPHA:1", "HGNC:3"),
        ("ORPHA:2", "HGNC:3"),
        ("ORPHA:4", "HGNC:3"),
    ]


def test_filtering_uses_known_positives_from_all_splits() -> None:
    package = load_graph_package(FIXTURE_DIR, seed=42)
    scorer = RecordingScorer()
    evaluate_link_prediction(scorer, package, "train")

    # Evaluating the TRAIN pair ORPHA:2/HGNC:2 must still filter out HGNC:3,
    # which is a VALIDATION positive for ORPHA:2.
    gene_batch, _ = _batches_for(scorer, ("ORPHA:2", "HGNC:2"))
    assert ("ORPHA:2", "HGNC:3") not in gene_batch
    assert sorted(gene_batch) == [
        ("ORPHA:2", "HGNC:1"),
        ("ORPHA:2", "HGNC:2"),
        ("ORPHA:2", "HGNC:4"),
    ]

    # Disease side for train pair ORPHA:3/HGNC:3: ORPHA:2 is a validation
    # positive for HGNC:3 and must be filtered out.
    _, disease_batch = _batches_for(scorer, ("ORPHA:3", "HGNC:3"))
    assert ("ORPHA:2", "HGNC:3") not in disease_batch


def test_filtered_known_positive_scored_higher_does_not_hurt_rank() -> None:
    package = load_graph_package(FIXTURE_DIR, seed=42)

    # Score the train positive ORPHA:2/HGNC:2 above everything; the validation
    # pair ORPHA:2/HGNC:3 must still rank first because HGNC:2 is filtered.
    def scorer(pairs: list[Pair]) -> list[float]:
        scores = {("ORPHA:2", "HGNC:2"): 100.0, ("ORPHA:2", "HGNC:3"): 50.0,
                  ("ORPHA:3", "HGNC:3"): 100.0}
        return [scores.get(pair, 0.0) for pair in pairs]

    report = evaluate_link_prediction(scorer, package, "validation")
    assert report.gene_corruption.hits_at_1 == 1.0
    assert report.disease_corruption.hits_at_1 == 1.0


def test_non_rankable_true_gene_stays_in_its_own_pool() -> None:
    package = load_graph_package(FIXTURE_DIR, seed=42)
    scorer = RecordingScorer()
    evaluate_link_prediction(scorer, package, "test")

    gene_batch, _ = _batches_for(scorer, ("ORPHA:1", "HGNC:5"))
    # HGNC:1 is a train positive for ORPHA:1 -> filtered. HGNC:5 is the true
    # gene (kept once even though non-rankable).
    assert sorted(gene_batch) == [
        ("ORPHA:1", "HGNC:2"),
        ("ORPHA:1", "HGNC:3"),
        ("ORPHA:1", "HGNC:4"),
        ("ORPHA:1", "HGNC:5"),
    ]


# --- metric arithmetic and tie policy ----------------------------------------


def test_constant_scorer_uses_mean_rank_tie_policy() -> None:
    package = load_graph_package(FIXTURE_DIR, seed=42)
    report = evaluate_link_prediction(RecordingScorer(), package, "validation")

    # Each direction pools 3 candidates, all tied -> mean rank (1+2+3)/3 = 2.
    expected = RankingMetrics(hits_at_1=0.0, hits_at_10=1.0, hits_at_50=1.0, mrr=0.5, count=1)
    assert report.gene_corruption == expected
    assert report.disease_corruption == expected
    assert report.combined == RankingMetrics(
        hits_at_1=0.0, hits_at_10=1.0, hits_at_50=1.0, mrr=0.5, count=2
    )


def test_worst_scorer_ranks_true_pair_last() -> None:
    package = load_graph_package(FIXTURE_DIR, seed=42)
    true_pairs = set(package.positives["validation"])

    def worst_scorer(pairs: list[Pair]) -> list[float]:
        return [-10.0 if pair in true_pairs else 10.0 for pair in pairs]

    report = evaluate_link_prediction(worst_scorer, package, "validation")
    assert report.combined.hits_at_1 == 0.0
    assert report.combined.mrr == pytest.approx(1.0 / 3.0)


def test_partial_tie_counts_strictly_greater_plus_half_ties() -> None:
    package = load_graph_package(FIXTURE_DIR, seed=42)

    # Gene pool for ORPHA:2/HGNC:3 is {HGNC:1, HGNC:3, HGNC:4}.
    # HGNC:1 strictly above, HGNC:4 tied -> rank = 1 + 1 + 0.5 = 2.5.
    def scorer(pairs: list[Pair]) -> list[float]:
        scores = {("ORPHA:2", "HGNC:1"): 5.0, ("ORPHA:2", "HGNC:3"): 1.0,
                  ("ORPHA:2", "HGNC:4"): 1.0}
        return [scores.get(pair, 0.0) for pair in pairs]

    report = evaluate_link_prediction(scorer, package, "validation")
    assert report.gene_corruption.mrr == pytest.approx(1.0 / 2.5)
    assert report.gene_corruption.hits_at_1 == 0.0
    assert report.gene_corruption.hits_at_10 == 1.0


# --- cold-start routing -------------------------------------------------------


def test_cold_start_routing_is_a_partition_in_both_directions() -> None:
    package = load_graph_package(FIXTURE_DIR, seed=42)
    for split in ("train", "validation", "test"):
        report = evaluate_link_prediction(RecordingScorer(), package, split)
        n = len(package.positives[split])
        assert report.gene_corruption.count + report.cold_start_gene_corruption.count == n
        assert (
            report.disease_corruption.count + report.cold_start_disease_corruption.count == n
        )
        assert report.combined.count == (
            report.gene_corruption.count + report.disease_corruption.count
        )
        # Full-set view merges main + cold-start buckets by construction.
        assert report.all_gene_corruption.count == (
            report.gene_corruption.count + report.cold_start_gene_corruption.count
        )
        assert report.all_disease_corruption.count == (
            report.disease_corruption.count + report.cold_start_disease_corruption.count
        )
        assert report.all_gene_corruption.count == n
        assert report.all_disease_corruption.count == n
        assert report.all_combined.count == 2 * n


def test_test_split_routes_only_gene_side_to_cold_start() -> None:
    package = load_graph_package(FIXTURE_DIR, seed=42)
    report = evaluate_link_prediction(RecordingScorer(), package, "test")
    # HGNC:5 has no edges -> cold-start gene; ORPHA:1 has edges -> main disease bucket.
    assert report.cold_start_gene_corruption.count == 1
    assert report.gene_corruption.count == 0
    assert report.disease_corruption.count == 1
    assert report.cold_start_disease_corruption.count == 0
    # combined never includes cold-start pairs
    assert report.combined.count == 1


def test_cold_start_metrics_are_computed_not_dropped() -> None:
    package = load_graph_package(FIXTURE_DIR, seed=42)

    def perfect(pairs: list[Pair]) -> list[float]:
        return [1.0 if pair == ("ORPHA:1", "HGNC:5") else 0.0 for pair in pairs]

    report = evaluate_link_prediction(perfect, package, "test")
    assert report.cold_start_gene_corruption.hits_at_1 == 1.0
    assert report.cold_start_gene_corruption.mrr == 1.0


# --- full-set (main + cold-start) view --------------------------------------


def test_all_metrics_merge_main_and_cold_start_ranks() -> None:
    package = load_graph_package(FIXTURE_DIR, seed=42)

    # Test pair ORPHA:1/HGNC:5: gene side is cold-start (HGNC:5 has no edges),
    # disease side is main. Gene candidates score below the true pair -> rank 1;
    # all 3 disease candidates (ORPHA:2/3/4 with HGNC:5) score above -> rank 4.
    def scorer(pairs: list[Pair]) -> list[float]:
        scores = {("ORPHA:1", "HGNC:5"): 1.0, ("ORPHA:2", "HGNC:5"): 5.0,
                  ("ORPHA:3", "HGNC:5"): 5.0, ("ORPHA:4", "HGNC:5"): 5.0}
        return [scores.get(pair, 0.0) for pair in pairs]

    report = evaluate_link_prediction(scorer, package, "test")
    # Main-only view sees just the disease-side rank 4.
    assert report.combined == RankingMetrics(
        hits_at_1=0.0, hits_at_10=1.0, hits_at_50=1.0, mrr=0.25, count=1
    )
    assert report.all_gene_corruption == report.cold_start_gene_corruption == RankingMetrics(
        hits_at_1=1.0, hits_at_10=1.0, hits_at_50=1.0, mrr=1.0, count=1
    )
    assert report.all_disease_corruption == report.disease_corruption
    # Full-set view averages both ranks {1, 4}.
    assert report.all_combined == RankingMetrics(
        hits_at_1=0.5, hits_at_10=1.0, hits_at_50=1.0, mrr=pytest.approx(0.625), count=2
    )


def test_all_metrics_equal_main_metrics_when_no_cold_start() -> None:
    package = load_graph_package(FIXTURE_DIR, seed=42)
    report = evaluate_link_prediction(RecordingScorer(), package, "validation")
    assert report.cold_start_gene_corruption.count == 0
    assert report.cold_start_disease_corruption.count == 0
    assert report.all_gene_corruption == report.gene_corruption
    assert report.all_disease_corruption == report.disease_corruption
    assert report.all_combined == report.combined


def _synthetic_package() -> GraphPackage:
    """Package where one disease has only a HAS_PHENOTYPE edge and one has nothing."""
    genes = build_node_index({
        "G1": {"rankable": True}, "G2": {"rankable": True}, "G3": {"rankable": True},
    })
    diseases = build_node_index({
        "D1": {"rankable": True},  # ASSOCIATED_WITH edge in train graph
        "D2": {"rankable": True},  # only HAS_PHENOTYPE edge -> NOT cold-start
        "D3": {"rankable": True},  # no edges at all -> cold-start
    })
    phenotypes = build_node_index({"P1": {"rankable": True}})
    node_indexes = {"Gene": genes, "Disease": diseases, "Phenotype": phenotypes}
    data = build_hetero_data(
        node_indexes,
        [
            {"source_id": "D1", "target_id": "G1", "relation": "ASSOCIATED_WITH"},
            {"source_id": "D2", "target_id": "P1", "relation": "HAS_PHENOTYPE"},
        ],
    )
    return GraphPackage(
        data=data,
        node_indexes=node_indexes,
        positives={
            "train": [("D1", "G1")],
            "validation": [],
            "test": [("D2", "G1"), ("D3", "G1")],
        },
        negatives={"train": [], "validation": [], "test": []},
    )


def test_disease_with_only_phenotype_edges_is_not_cold_start() -> None:
    package = _synthetic_package()
    report = evaluate_link_prediction(RecordingScorer(), package, "test")
    # D2 has a HAS_PHENOTYPE edge -> main; D3 has none -> cold-start.
    assert report.disease_corruption.count == 1
    assert report.cold_start_disease_corruption.count == 1
    # G1 has a train ASSOCIATED_WITH edge -> both pairs in main gene bucket.
    assert report.gene_corruption.count == 2
    assert report.cold_start_gene_corruption.count == 0


def test_empty_bucket_reports_zero_count() -> None:
    package = load_graph_package(FIXTURE_DIR, seed=42)
    report = evaluate_link_prediction(RecordingScorer(), package, "validation")
    assert report.cold_start_gene_corruption == RankingMetrics(
        hits_at_1=0.0, hits_at_10=0.0, hits_at_50=0.0, mrr=0.0, count=0
    )


# --- input validation ---------------------------------------------------------


def test_unknown_split_raises() -> None:
    package = load_graph_package(FIXTURE_DIR, seed=42)
    with pytest.raises(KeyError):
        evaluate_link_prediction(RecordingScorer(), package, "holdout")


def test_scorer_returning_wrong_length_raises() -> None:
    package = load_graph_package(FIXTURE_DIR, seed=42)

    def bad_scorer(pairs: list[Pair]) -> list[float]:
        return [0.0]

    with pytest.raises(ValueError):
        evaluate_link_prediction(bad_scorer, package, "validation")


def test_scorer_returning_nan_for_true_pair_raises() -> None:
    package = load_graph_package(FIXTURE_DIR, seed=42)
    true_pairs = set(package.positives["validation"])

    def nan_for_true_pair_scorer(pairs: list[Pair]) -> list[float]:
        return [float("nan") if pair in true_pairs else 0.0 for pair in pairs]

    # Without the finiteness check, a NaN true-pair score would silently
    # compute to rank 1 (nan > x and nan == x are both False), reporting a
    # perfect Hits@1/MRR for a completely broken scorer.
    with pytest.raises(ValueError, match="non-finite"):
        evaluate_link_prediction(nan_for_true_pair_scorer, package, "validation")


def test_scorer_returning_nan_for_a_candidate_raises() -> None:
    package = load_graph_package(FIXTURE_DIR, seed=42)

    def nan_for_one_candidate_scorer(pairs: list[Pair]) -> list[float]:
        # Poison exactly one candidate (not the true pair, which is pairs[0]).
        return [0.0 if index == 0 else float("nan") for index, _ in enumerate(pairs)]

    with pytest.raises(ValueError, match="non-finite"):
        evaluate_link_prediction(nan_for_one_candidate_scorer, package, "validation")


# --- candidate-subset ranking (small, disease-specific gene pool) ------------
#
# Fixture genes-by-disease (all splits): ORPHA:1 -> {HGNC:1, HGNC:5},
# ORPHA:2 -> {HGNC:2, HGNC:3}, ORPHA:3 -> {HGNC:3}, ORPHA:4 -> {HGNC:4}.
# Rankable genes: HGNC:1-4 (HGNC:5 is not rankable).


def test_candidate_subset_pool_uses_other_known_genes_for_same_disease() -> None:
    package = load_graph_package(FIXTURE_DIR, seed=42)
    scorer = RecordingScorer()
    # validation pair (ORPHA:2, HGNC:3); ORPHA:2's only other known gene is
    # HGNC:2, so with pool_size=2 the pool is exactly {HGNC:2}, no padding.
    evaluate_candidate_subset_ranking(scorer, package, "validation", pool_size=2)
    assert len(scorer.batches) == 1
    assert sorted(scorer.batches[0]) == [("ORPHA:2", "HGNC:2"), ("ORPHA:2", "HGNC:3")]


def test_candidate_subset_pads_with_rankable_genes_when_known_is_insufficient() -> None:
    package = load_graph_package(FIXTURE_DIR, seed=42)
    scorer = RecordingScorer()
    # test pair (ORPHA:1, HGNC:5); only other known gene is HGNC:1, so a
    # pool_size=4 pool (3 candidates) needs 2 padded genes from {HGNC:2,3,4}.
    evaluate_candidate_subset_ranking(scorer, package, "test", pool_size=4, seed=42)
    batch = scorer.batches[0]
    assert len(batch) == 4
    assert ("ORPHA:1", "HGNC:5") in batch
    assert ("ORPHA:1", "HGNC:1") in batch
    padded = {gene for _, gene in batch} - {"HGNC:5", "HGNC:1"}
    assert padded <= {"HGNC:2", "HGNC:3", "HGNC:4"}
    assert len(padded) == 2


def test_candidate_subset_padding_is_deterministic_for_fixed_seed() -> None:
    package = load_graph_package(FIXTURE_DIR, seed=42)
    scorer_a, scorer_b = RecordingScorer(), RecordingScorer()
    evaluate_candidate_subset_ranking(scorer_a, package, "test", pool_size=4, seed=7)
    evaluate_candidate_subset_ranking(scorer_b, package, "test", pool_size=4, seed=7)
    assert scorer_a.batches == scorer_b.batches


def test_candidate_subset_perfect_scorer_ranks_true_gene_first() -> None:
    package = load_graph_package(FIXTURE_DIR, seed=42)
    true_pairs = set(package.positives["validation"])

    def perfect(pairs: list[Pair]) -> list[float]:
        return [10.0 if pair in true_pairs else -10.0 for pair in pairs]

    metrics = evaluate_candidate_subset_ranking(perfect, package, "validation")
    assert metrics.hits_at_1 == 1.0
    assert metrics.mrr == 1.0


def test_candidate_subset_reports_mean_known_candidates() -> None:
    package = load_graph_package(FIXTURE_DIR, seed=42)
    # validation has one pair (ORPHA:2, HGNC:3) with exactly one other known
    # gene (HGNC:2) for ORPHA:2.
    metrics = evaluate_candidate_subset_ranking(RecordingScorer(), package, "validation")
    assert metrics.mean_known_candidates == 1.0


def test_candidate_subset_rejects_non_positive_pool_size() -> None:
    package = load_graph_package(FIXTURE_DIR, seed=42)
    with pytest.raises(ValueError):
        evaluate_candidate_subset_ranking(RecordingScorer(), package, "validation", pool_size=0)


def test_candidate_subset_empty_split_reports_zero_count() -> None:
    package = load_graph_package(FIXTURE_DIR, seed=42)
    metrics = evaluate_candidate_subset_ranking(RecordingScorer(), package, "train", pool_size=1)
    # pool_size=1 keeps only the true gene -> every pair ranks first trivially.
    assert metrics.count == len(package.positives["train"])
    assert metrics.hits_at_1 == 1.0

    empty_package = load_graph_package(FIXTURE_DIR, seed=42)
    empty_package.positives["validation"] = []
    metrics = evaluate_candidate_subset_ranking(RecordingScorer(), empty_package, "validation")
    assert metrics == CandidateSubsetMetrics(
        hits_at_1=0.0, hits_at_3=0.0, hits_at_10=0.0, mrr=0.0, count=0,
        mean_known_candidates=0.0,
    )


def test_candidate_subset_unknown_split_raises() -> None:
    package = load_graph_package(FIXTURE_DIR, seed=42)
    with pytest.raises(KeyError):
        evaluate_candidate_subset_ranking(RecordingScorer(), package, "holdout")


# --- multi-disease candidate pool (Stage A+B+C end-to-end stand-in) ----------
#
# Fixture genes-by-disease (all splits): ORPHA:1 -> {HGNC:1, HGNC:5},
# ORPHA:2 -> {HGNC:2, HGNC:3}, ORPHA:3 -> {HGNC:3}, ORPHA:4 -> {HGNC:4}.
# Gene connectivity (# known diseases): HGNC:1=1, HGNC:2=1, HGNC:3=2 (ORPHA:2
# and ORPHA:3), HGNC:4=1, HGNC:5=1. Rankable genes: HGNC:1-4.


def test_multi_disease_single_disease_pool_matches_known_genes() -> None:
    package = load_graph_package(FIXTURE_DIR, seed=42)
    scorer = RecordingScorer()
    # num_diseases=1 means no Top-N companions; same union as the
    # candidate-subset pool for ORPHA:2 alone: {HGNC:2, HGNC:3}.
    evaluate_multi_disease_candidate_pool(
        scorer, package, "validation", num_diseases=1, pool_size=2
    )
    assert len(scorer.batches) == 1
    assert sorted(scorer.batches[0]) == [("ORPHA:2", "HGNC:2"), ("ORPHA:2", "HGNC:3")]


def test_multi_disease_pads_with_rankable_genes_when_union_is_small() -> None:
    package = load_graph_package(FIXTURE_DIR, seed=42)
    scorer = RecordingScorer()
    evaluate_multi_disease_candidate_pool(
        scorer, package, "validation", num_diseases=1, pool_size=4, seed=42
    )
    batch = scorer.batches[0]
    assert len(batch) == 4
    assert ("ORPHA:2", "HGNC:3") in batch
    padded = {gene for _, gene in batch} - {"HGNC:2", "HGNC:3"}
    assert padded <= {"HGNC:1", "HGNC:4"}
    assert len(padded) == 2


def test_multi_disease_truncation_can_drop_true_gene_as_stage_b_miss() -> None:
    package = load_graph_package(FIXTURE_DIR, seed=42)
    scorer = RecordingScorer()
    # num_diseases=4 unions every disease's known genes into one pool of 5
    # ({HGNC:1..5}); pool_size=1 keeps only the top-connectivity gene,
    # HGNC:3 (connectivity 2). Of the 4 train pairs, only (ORPHA:3, HGNC:3)
    # survives -- the other 3 are Stage B misses and never reach score_pair.
    metrics = evaluate_multi_disease_candidate_pool(
        scorer, package, "train", num_diseases=4, pool_size=1
    )
    assert metrics.count == 4
    assert metrics.pool_recall == 0.25
    assert metrics.mean_pool_size == 5.0
    assert metrics.hits_at_1 == 0.25
    assert metrics.hits_at_3 == 0.25
    assert metrics.hits_at_10 == 0.25
    assert metrics.mrr == 0.25
    assert scorer.batches == [[("ORPHA:3", "HGNC:3")]]


def test_multi_disease_padding_is_deterministic_for_fixed_seed() -> None:
    package = load_graph_package(FIXTURE_DIR, seed=42)
    scorer_a, scorer_b = RecordingScorer(), RecordingScorer()
    evaluate_multi_disease_candidate_pool(scorer_a, package, "test", pool_size=4, seed=7)
    evaluate_multi_disease_candidate_pool(scorer_b, package, "test", pool_size=4, seed=7)
    assert scorer_a.batches == scorer_b.batches


def test_multi_disease_perfect_scorer_ranks_true_gene_first_when_no_truncation() -> None:
    package = load_graph_package(FIXTURE_DIR, seed=42)
    true_pairs = set(package.positives["validation"])

    def perfect(pairs: list[Pair]) -> list[float]:
        return [10.0 if pair in true_pairs else -10.0 for pair in pairs]

    # Default pool_size=10 never truncates this 5-gene universe, so every
    # pair survives Stage B.
    metrics = evaluate_multi_disease_candidate_pool(perfect, package, "validation")
    assert metrics.pool_recall == 1.0
    assert metrics.hits_at_1 == 1.0
    assert metrics.mrr == 1.0


def test_multi_disease_rejects_non_positive_pool_size() -> None:
    package = load_graph_package(FIXTURE_DIR, seed=42)
    with pytest.raises(ValueError):
        evaluate_multi_disease_candidate_pool(
            RecordingScorer(), package, "validation", pool_size=0
        )


def test_multi_disease_rejects_non_positive_num_diseases() -> None:
    package = load_graph_package(FIXTURE_DIR, seed=42)
    with pytest.raises(ValueError):
        evaluate_multi_disease_candidate_pool(
            RecordingScorer(), package, "validation", num_diseases=0
        )


def test_multi_disease_empty_split_reports_zero_count() -> None:
    empty_package = load_graph_package(FIXTURE_DIR, seed=42)
    empty_package.positives["validation"] = []
    metrics = evaluate_multi_disease_candidate_pool(RecordingScorer(), empty_package, "validation")
    assert metrics == MultiDiseaseCandidateMetrics(
        hits_at_1=0.0, hits_at_3=0.0, hits_at_10=0.0, mrr=0.0, count=0,
        pool_recall=0.0, mean_pool_size=0.0,
    )


def test_multi_disease_unknown_split_raises() -> None:
    package = load_graph_package(FIXTURE_DIR, seed=42)
    with pytest.raises(KeyError):
        evaluate_multi_disease_candidate_pool(RecordingScorer(), package, "holdout")
