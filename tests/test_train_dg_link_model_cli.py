"""Tests for the `train_dg_link_model` CLI entry point."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from scripts.train_dg_link_model import main

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "modeling"


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
