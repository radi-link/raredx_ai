"""Tests for multi-seed train+evaluate aggregation."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from app.modeling.evaluate import EvaluationReport, RankingMetrics
from app.modeling.loader import load_graph_package
from app.modeling.multi_seed import (
    EVALUATION_REPORT_FIELDS,
    EvaluationReportStats,
    aggregate_evaluation_reports,
    aggregate_ranking_metrics,
    run_baseline_multi_seed,
    run_gnn_multi_seed,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "modeling"


def _metrics(mrr: float, count: int = 10) -> RankingMetrics:
    return RankingMetrics(hits_at_1=mrr, hits_at_10=mrr, hits_at_50=mrr, mrr=mrr, count=count)


def test_evaluation_report_fields_derived_from_dataclass_not_hardcoded() -> None:
    assert EVALUATION_REPORT_FIELDS == tuple(f.name for f in dataclasses.fields(EvaluationReport))


def test_evaluation_report_stats_covers_the_same_fields_as_evaluation_report() -> None:
    stats_fields = {f.name for f in dataclasses.fields(EvaluationReportStats)}
    assert stats_fields == set(EVALUATION_REPORT_FIELDS)


def test_aggregate_ranking_metrics_computes_mean_and_std() -> None:
    stats = aggregate_ranking_metrics([_metrics(0.1), _metrics(0.2), _metrics(0.3)])
    assert stats.mrr.mean == pytest.approx(0.2)
    assert stats.mrr.std == pytest.approx(0.1)
    assert stats.mrr.values == [0.1, 0.2, 0.3]
    assert stats.count == 10


def test_aggregate_ranking_metrics_single_value_has_zero_std() -> None:
    stats = aggregate_ranking_metrics([_metrics(0.5)])
    assert stats.mrr.mean == 0.5
    assert stats.mrr.std == 0.0


def test_aggregate_ranking_metrics_rejects_empty_list() -> None:
    with pytest.raises(ValueError, match="at least one"):
        aggregate_ranking_metrics([])


def test_aggregate_ranking_metrics_rejects_mismatched_counts() -> None:
    with pytest.raises(ValueError, match="inconsistent counts"):
        aggregate_ranking_metrics([_metrics(0.1, count=10), _metrics(0.2, count=20)])


def test_aggregate_evaluation_reports_covers_every_bucket() -> None:
    package = load_graph_package(FIXTURE_DIR, seed=42)
    run = run_baseline_multi_seed(package, [1, 2], epochs=3, patience=None)
    stats = aggregate_evaluation_reports(list(run.validation_per_seed.values()))
    assert stats.combined.mrr.values == [
        run.validation_per_seed[1].combined.mrr,
        run.validation_per_seed[2].combined.mrr,
    ]
    assert stats.all_combined.count == run.validation_per_seed[1].all_combined.count


def test_run_baseline_multi_seed_trains_once_per_seed() -> None:
    package = load_graph_package(FIXTURE_DIR, seed=42)
    run = run_baseline_multi_seed(package, [1, 2, 3], epochs=3, patience=None)
    assert set(run.validation_per_seed) == {1, 2, 3}
    assert set(run.test_per_seed) == {1, 2, 3}
    assert set(run.histories) == {1, 2, 3}
    for history in run.histories.values():
        assert len(history.train_loss) == 3
    assert run.validation_summary.combined.mrr.values != run.test_summary.combined.mrr.values


def test_run_baseline_multi_seed_different_seeds_give_different_histories() -> None:
    package = load_graph_package(FIXTURE_DIR, seed=42)
    run = run_baseline_multi_seed(package, [1, 2], epochs=5, patience=None)
    assert run.histories[1].train_loss != run.histories[2].train_loss


def test_run_gnn_multi_seed_trains_once_per_seed() -> None:
    package = load_graph_package(FIXTURE_DIR, seed=42)
    run = run_gnn_multi_seed(
        package, [1, 2], epochs=3, hidden_dim=8, embedding_dim=8, patience=None
    )
    assert set(run.validation_per_seed) == {1, 2}
    assert set(run.histories) == {1, 2}


def test_run_baseline_multi_seed_calls_on_seed_done_once_per_seed_as_it_goes() -> None:
    package = load_graph_package(FIXTURE_DIR, seed=42)
    calls: list[tuple[int, EvaluationReport, EvaluationReport]] = []

    def on_seed_done(seed, model, validation_report, test_report, history) -> None:  # type: ignore[no-untyped-def]
        calls.append((seed, validation_report, test_report))

    run = run_baseline_multi_seed(
        package, [1, 2, 3], epochs=3, patience=None, on_seed_done=on_seed_done
    )
    assert [c[0] for c in calls] == [1, 2, 3]
    assert calls[0][1] is run.validation_per_seed[1]
    assert calls[-1][2] is run.test_per_seed[3]


def test_run_multi_seed_rejects_empty_seed_list() -> None:
    package = load_graph_package(FIXTURE_DIR, seed=42)
    with pytest.raises(ValueError, match="non-empty"):
        run_baseline_multi_seed(package, [], epochs=3)
