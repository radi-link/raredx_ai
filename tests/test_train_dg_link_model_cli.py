"""Tests for the `train_dg_link_model` CLI entry point."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from scripts.train_dg_link_model import main

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "modeling"


def _write_stage_a_processed_store(processed_dir: Path) -> None:
    """Write the three processed-store files needed for IC Stage A queries."""
    processed_dir.mkdir()
    (processed_dir / "phenotypes.json").write_text(
        json.dumps(
            [
                {"hpo_id": "HP:0000001", "name": "Phenotype one"},
                {"hpo_id": "HP:0000002", "name": "Phenotype two"},
                {"hpo_id": "HP:0000003", "name": "Phenotype three"},
            ]
        ),
        encoding="utf-8",
    )
    (processed_dir / "disease_phenotypes.json").write_text(
        json.dumps(
            [
                {"disease_id": "ORPHA:1", "disease_name": "Disease1", "hpo_id": "HP:0000001"},
                {"disease_id": "ORPHA:2", "disease_name": "Disease2", "hpo_id": "HP:0000002"},
                {"disease_id": "ORPHA:3", "disease_name": "Disease3", "hpo_id": "HP:0000003"},
                {"disease_id": "ORPHA:4", "disease_name": "Disease4", "hpo_id": "HP:0000001"},
            ]
        ),
        encoding="utf-8",
    )
    (processed_dir / "gene_phenotypes.json").write_text(
        json.dumps(
            [
                {
                    "gene_id": "HGNC:1",
                    "gene_symbol": "HGNC:1",
                    "hpo_id": "HP:0000001",
                    "disease_id": "ORPHA:1",
                }
            ]
        ),
        encoding="utf-8",
    )


def test_cli_omits_stage_a_connected_section_without_processed_dir(tmp_path: Path) -> None:
    """A missing opt-in must retain the established report shape exactly."""
    output = tmp_path / "report.json"

    assert main(
        [
            "--graph-dir",
            str(FIXTURE_DIR),
            "--epochs",
            "2",
            "--output",
            str(output),
            "--model-dir",
            str(tmp_path / "models"),
        ]
    ) == 0

    report = json.loads(output.read_text())
    assert set(report) == {
        "training",
        "baseline",
        "gnn",
        "candidate_subset",
        "multi_disease_candidate_pool",
    }


def test_cli_writes_ic_stage_a_connected_metrics_and_limitations(tmp_path: Path) -> None:
    """The IC opt-in emits pipeline metrics and explicitly scopes their meaning."""
    output = tmp_path / "report.json"
    processed_dir = tmp_path / "processed"
    _write_stage_a_processed_store(processed_dir)

    assert main(
        [
            "--graph-dir",
            str(FIXTURE_DIR),
            "--epochs",
            "2",
            "--stage-a-processed-dir",
            str(processed_dir),
            "--stage-a-method",
            "ic",
            "--output",
            str(output),
            "--model-dir",
            str(tmp_path / "models"),
        ]
    ) == 0

    report = json.loads(output.read_text())
    assert set(report) == {
        "training",
        "baseline",
        "gnn",
        "candidate_subset",
        "multi_disease_candidate_pool",
        "config",
        "stage_a_connected_candidate_pool",
    }
    for model in ("baseline", "gnn"):
        for split in ("validation", "test"):
            metrics = report["stage_a_connected_candidate_pool"][model][split]
            assert set(metrics) == {
                "hits_at_1",
                "hits_at_3",
                "hits_at_10",
                "mrr",
                "count",
                "skipped_no_query_count",
                "stage_a_recall_at_n",
                "stage_b_recall_given_stage_a",
                "end_to_end_pool_recall",
                "mean_stage_a_candidate_count",
                "mean_pool_size",
            }
    metadata = report["config"]["stage_a_connected_evaluation"]
    assert metadata["method"] == "ic"
    assert metadata["aggregation"] == "max_predicted_disease_score"
    assert metadata["query_protocol"] == "synthetic_processed_disease_hpo_profile"
    assert "not an independent clinical benchmark" in metadata["limitation"]


def test_cli_rejects_stage_a_method_without_processed_dir(tmp_path: Path) -> None:
    """A Stage A method cannot silently run without its processed knowledge base."""
    with pytest.raises(ValueError, match="--stage-a-method requires --stage-a-processed-dir"):
        main(
            [
                "--graph-dir",
                str(FIXTURE_DIR),
                "--stage-a-method",
                "ic",
                "--output",
                str(tmp_path / "report.json"),
                "--model-dir",
                str(tmp_path / "models"),
            ]
        )


def test_cli_writes_report_with_expected_keys(tmp_path: Path) -> None:
    output = tmp_path / "report.json"
    model_dir = tmp_path / "models"
    exit_code = main(
        [
            "--graph-dir",
            str(FIXTURE_DIR),
            "--seed",
            "42",
            "--epochs",
            "3",
            "--output",
            str(output),
            "--model-dir",
            str(model_dir),
        ]
    )
    assert exit_code == 0
    report = json.loads(output.read_text())
    assert set(report) == {
        "training",
        "baseline",
        "gnn",
        "candidate_subset",
        "multi_disease_candidate_pool",
    }
    assert set(report["training"]) == {"baseline", "gnn"}
    for model in ("baseline", "gnn"):
        training = report["training"][model]
        assert training["epochs_run"] == 3
        assert 1 <= training["best_epoch"] <= 3
    assert set(report["baseline"]) == {"validation", "test"}
    assert set(report["gnn"]) == {"validation", "test"}
    assert "hits_at_1" in report["baseline"]["test"]["combined"]
    assert "hits_at_1" in report["gnn"]["test"]["combined"]
    for model in ("baseline", "gnn"):
        for split in ("validation", "test"):
            split_report = report[model][split]
            for key in ("all_gene_corruption", "all_disease_corruption", "all_combined"):
                assert key in split_report
            assert "hits_at_1" in split_report["all_combined"]
            assert split_report["all_gene_corruption"]["count"] == (
                split_report["gene_corruption"]["count"]
                + split_report["cold_start_gene_corruption"]["count"]
            )


def test_cli_writes_candidate_subset_section(tmp_path: Path) -> None:
    output = tmp_path / "report.json"
    model_dir = tmp_path / "models"
    exit_code = main(
        [
            "--graph-dir",
            str(FIXTURE_DIR),
            "--seed",
            "42",
            "--epochs",
            "3",
            "--candidate-pool-size",
            "2",
            "--output",
            str(output),
            "--model-dir",
            str(model_dir),
        ]
    )
    assert exit_code == 0
    report = json.loads(output.read_text())
    assert set(report["candidate_subset"]) == {"baseline", "gnn"}
    for model in ("baseline", "gnn"):
        for split in ("validation", "test"):
            metrics = report["candidate_subset"][model][split]
            assert set(metrics) == {
                "hits_at_1",
                "hits_at_3",
                "hits_at_10",
                "mrr",
                "count",
                "mean_known_candidates",
            }


def test_cli_writes_multi_disease_candidate_pool_section(tmp_path: Path) -> None:
    output = tmp_path / "report.json"
    model_dir = tmp_path / "models"
    exit_code = main(
        [
            "--graph-dir",
            str(FIXTURE_DIR),
            "--seed",
            "42",
            "--epochs",
            "3",
            "--candidate-pool-size",
            "2",
            "--num-candidate-diseases",
            "2",
            "--output",
            str(output),
            "--model-dir",
            str(model_dir),
        ]
    )
    assert exit_code == 0
    report = json.loads(output.read_text())
    assert set(report["multi_disease_candidate_pool"]) == {"baseline", "gnn"}
    for model in ("baseline", "gnn"):
        for split in ("validation", "test"):
            metrics = report["multi_disease_candidate_pool"][model][split]
            assert set(metrics) == {
                "hits_at_1",
                "hits_at_3",
                "hits_at_10",
                "mrr",
                "count",
                "pool_recall",
                "mean_pool_size",
            }


def test_cli_creates_output_directory_if_missing(tmp_path: Path) -> None:
    output = tmp_path / "nested" / "dir" / "report.json"
    model_dir = tmp_path / "models"
    exit_code = main(
        [
            "--graph-dir",
            str(FIXTURE_DIR),
            "--seed",
            "42",
            "--epochs",
            "2",
            "--output",
            str(output),
            "--model-dir",
            str(model_dir),
        ]
    )
    assert exit_code == 0
    assert output.is_file()


def test_cli_saves_loadable_model_checkpoints(tmp_path: Path) -> None:
    output = tmp_path / "report.json"
    model_dir = tmp_path / "models"
    exit_code = main(
        [
            "--graph-dir",
            str(FIXTURE_DIR),
            "--seed",
            "42",
            "--epochs",
            "2",
            "--output",
            str(output),
            "--model-dir",
            str(model_dir),
        ]
    )
    assert exit_code == 0
    baseline_checkpoint = model_dir / "baseline_state_dict.pt"
    gnn_checkpoint = model_dir / "gnn_state_dict.pt"
    assert baseline_checkpoint.is_file()
    assert gnn_checkpoint.is_file()
    baseline_state = torch.load(baseline_checkpoint, weights_only=True)
    gnn_state = torch.load(gnn_checkpoint, weights_only=True)
    assert "gene_prior" in baseline_state
    assert any(key.startswith("relation_weight") for key in gnn_state)


def test_cli_patience_stops_training_before_the_full_epoch_budget(tmp_path: Path) -> None:
    output = tmp_path / "report.json"
    model_dir = tmp_path / "models"
    exit_code = main(
        [
            "--graph-dir",
            str(FIXTURE_DIR),
            "--seed",
            "42",
            "--epochs",
            "200",
            "--patience",
            "3",
            "--output",
            str(output),
            "--model-dir",
            str(model_dir),
        ]
    )
    assert exit_code == 0
    report = json.loads(output.read_text())
    for model in ("baseline", "gnn"):
        training = report["training"][model]
        assert training["epochs_run"] < 200
        assert training["epochs_run"] == training["best_epoch"] + 3


def test_cli_patience_zero_disables_early_stopping(tmp_path: Path) -> None:
    output = tmp_path / "report.json"
    model_dir = tmp_path / "models"
    exit_code = main(
        [
            "--graph-dir",
            str(FIXTURE_DIR),
            "--seed",
            "42",
            "--epochs",
            "4",
            "--patience",
            "0",
            "--output",
            str(output),
            "--model-dir",
            str(model_dir),
        ]
    )
    assert exit_code == 0
    report = json.loads(output.read_text())
    for model in ("baseline", "gnn"):
        assert report["training"][model]["epochs_run"] == 4


def test_cli_refuses_to_overwrite_existing_outputs_without_flag(tmp_path: Path) -> None:
    output = tmp_path / "report.json"
    model_dir = tmp_path / "models"
    args = [
        "--graph-dir",
        str(FIXTURE_DIR),
        "--seed",
        "42",
        "--epochs",
        "2",
        "--output",
        str(output),
        "--model-dir",
        str(model_dir),
    ]
    assert main(args) == 0

    with pytest.raises(FileExistsError):
        main(args)

    assert main([*args, "--overwrite"]) == 0
