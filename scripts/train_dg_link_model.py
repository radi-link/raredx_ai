"""CLI: train baseline and GNN disease-gene link predictors, write a JSON report.

Trains `BaselineScorer` and `GNNLinkPredictor` (see `app/modeling/train.py`) on a
graph package (see `app/modeling/loader.py`) once per seed in `--model-seeds`
(same fixed graph split throughout), evaluates each run on the validation and
test splits via `app/modeling/multi_seed.py`, and writes a JSON report with
top-level keys `"status"`, `"config"`, `"training"`, `"baseline"`, `"gnn"` -- the
latter two each holding `{"validation": {...}, "test": {...}}` with a `"summary"`
(mean/std across seeds) and a `"per_seed"` breakdown.

A single seed's Hits@K/MRR can swing well beyond the differences between
hyperparameter choices (see `docs/data/modeling_results_v3.md`), so this script
always reports multi-seed statistics -- `--model-seeds 42` (the default) just
makes that a one-seed summary with std 0.0, not a different code path.

The report and each (model, seed) checkpoint are written incrementally, right
after that seed finishes -- not only once at the very end -- so a crash partway
through a long `--model-seeds 1,2,3,4,5` run loses only the seed in progress,
not every already-completed one. `"status"` is `"in_progress"` until every seed
of both models has finished, then `"complete"`.

Refuses to overwrite an existing checkpoint or report file unless `--overwrite`
is passed; `--overwrite` also removes any stale `*_state_dict_seed*.pt` left in
`--model-dir` by an earlier run under a different `--model-seeds`/config, so old
and new checkpoints never end up silently mixed in the same directory.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
from torch import nn

from app.modeling.evaluate import EvaluationReport
from app.modeling.loader import GraphPackage, load_graph_package
from app.modeling.multi_seed import (
    aggregate_evaluation_reports,
    run_baseline_multi_seed,
    run_gnn_multi_seed,
)
from app.modeling.train import TrainingHistory

MODEL_NAMES: tuple[str, str] = ("baseline", "gnn")


def _parse_seeds(raw: str) -> list[int]:
    seeds = [int(piece.strip()) for piece in raw.split(",") if piece.strip()]
    if not seeds:
        raise ValueError(f"--model-seeds produced no seeds from {raw!r}")
    duplicates = {seed for seed in seeds if seeds.count(seed) > 1}
    if duplicates:
        raise ValueError(
            f"--model-seeds has duplicate seed(s) {sorted(duplicates)}: {raw!r} -- each "
            "seed trains a model, so a repeat wastes compute and would silently collapse "
            "in per-seed results while the reported seed count stayed wrong"
        )
    return seeds


def _check_no_clobber(paths: list[Path], *, overwrite: bool) -> None:
    if overwrite:
        return
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise FileExistsError(
            "refusing to overwrite existing output file(s) without --overwrite: "
            + ", ".join(existing)
        )


def _clean_stale_checkpoints(model_dir: Path, keep: set[Path]) -> None:
    """Remove `*_state_dict_seed*.pt` checkpoints under `model_dir` not in `keep`.

    Only called under `--overwrite`. Without this, a later run with a different
    `--model-seeds` (or a different model-dir reused across configs) would leave a
    prior run's checkpoints sitting alongside the new ones with no warning --
    anyone loading "every checkpoint in model-dir" would silently mix models
    trained under different configs.
    """
    if not model_dir.exists():
        return
    for existing in model_dir.glob("*_state_dict_seed*.pt"):
        if existing not in keep:
            existing.unlink()


class _IncrementalReport:
    """Accumulates per-seed results as they finish and writes the report after each one.

    Also saves that seed's checkpoint immediately. Together these mean a crash
    partway through a multi-seed run loses at most the one seed in progress, not
    every seed already completed -- see this module's docstring.
    """

    def __init__(self, output: Path, model_dir: Path, config: dict) -> None:
        self._output = output
        self._model_dir = model_dir
        self._config = config
        self._validation: dict[str, dict[int, EvaluationReport]] = {name: {} for name in MODEL_NAMES}
        self._test: dict[str, dict[int, EvaluationReport]] = {name: {} for name in MODEL_NAMES}
        self._training: dict[str, dict[str, dict]] = {name: {} for name in MODEL_NAMES}
        self._expected_seed_count: dict[str, int] = {name: 0 for name in MODEL_NAMES}

    def expect(self, model_name: str, seed_count: int) -> None:
        self._expected_seed_count[model_name] = seed_count

    def make_callback(self, model_name: str):
        def on_seed_done(
            seed: int,
            model: nn.Module,
            validation_report: EvaluationReport,
            test_report: EvaluationReport,
            history: TrainingHistory,
        ) -> None:
            self._model_dir.mkdir(parents=True, exist_ok=True)
            checkpoint = self._model_dir / f"{model_name}_state_dict_seed{seed}.pt"
            torch.save(model.state_dict(), checkpoint)

            self._validation[model_name][seed] = validation_report
            self._test[model_name][seed] = test_report
            self._training[model_name][str(seed)] = {
                "epochs_run": len(history.train_loss),
                "best_epoch": history.best_epoch,
                "best_validation_loss": min(history.validation_loss),
            }
            self._write()

        return on_seed_done

    def _split_to_dict(self, model_name: str, by_split: dict[str, dict[int, EvaluationReport]]) -> dict:
        reports = by_split[model_name]
        if not reports:
            return {"summary": None, "per_seed": {}}
        summary = aggregate_evaluation_reports(list(reports.values()))
        return {
            "summary": dataclasses.asdict(summary),
            "per_seed": {str(seed): dataclasses.asdict(r) for seed, r in reports.items()},
        }

    def _status(self) -> str:
        done = all(
            len(self._validation[name]) >= self._expected_seed_count[name]
            for name in MODEL_NAMES
        )
        return "complete" if done else "in_progress"

    def _write(self) -> None:
        report = {
            "status": self._status(),
            "config": self._config,
            "training": {name: self._training[name] for name in MODEL_NAMES},
            **{
                name: {
                    "validation": self._split_to_dict(name, self._validation),
                    "test": self._split_to_dict(name, self._test),
                }
                for name in MODEL_NAMES
            },
        }
        self._output.parent.mkdir(parents=True, exist_ok=True)
        self._output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


def _run_all(
    package: GraphPackage,
    model_seeds: list[int],
    *,
    epochs: int,
    weight_decay: float,
    patience: int | None,
    incremental: _IncrementalReport,
) -> None:
    incremental.expect("baseline", len(model_seeds))
    incremental.expect("gnn", len(model_seeds))
    run_baseline_multi_seed(
        package,
        model_seeds,
        epochs=epochs,
        weight_decay=weight_decay,
        patience=patience,
        on_seed_done=incremental.make_callback("baseline"),
    )
    run_gnn_multi_seed(
        package,
        model_seeds,
        epochs=epochs,
        weight_decay=weight_decay,
        patience=patience,
        on_seed_done=incremental.make_callback("gnn"),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Train baseline and GNN disease-gene link prediction models across one "
            "or more model seeds and write a filtered, bidirectional Hits@K/MRR "
            "evaluation report with per-seed and mean/std summary statistics."
        )
    )
    parser.add_argument("--graph-dir", type=Path, default=Path("data/processed/graph"))
    parser.add_argument(
        "--seed", type=int, default=42, help="seed for the graph package's train/val/test split"
    )
    parser.add_argument(
        "--model-seeds",
        type=str,
        default="42",
        help=(
            "comma-separated model seeds to train+evaluate, e.g. '1,2,3,4,5'; the graph "
            "split (--seed) stays fixed across all of them. Report always includes "
            "per-seed results plus a mean/std summary, even for a single seed."
        ),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=400,
        help="epoch budget; with --patience set (the default), training usually stops earlier",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=20,
        help=(
            "stop training after this many epochs with no new best validation loss, and "
            "restore the best epoch's weights; pass 0 to disable early stopping and always "
            "train for the full --epochs, returning the final epoch's weights"
        ),
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="Adam L2 penalty for both models; pass 0 to disable",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("reports/dg_link_prediction_report.json")
    )
    parser.add_argument("--model-dir", type=Path, default=Path("reports/models"))
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="allow overwriting an existing report or checkpoint at the output paths",
    )
    args = parser.parse_args(argv)
    patience = None if args.patience <= 0 else args.patience
    model_seeds = _parse_seeds(args.model_seeds)

    checkpoint_paths = [
        args.model_dir / f"{model_name}_state_dict_seed{seed}.pt"
        for model_name in MODEL_NAMES
        for seed in model_seeds
    ]
    _check_no_clobber([args.output, *checkpoint_paths], overwrite=args.overwrite)
    if args.overwrite:
        _clean_stale_checkpoints(args.model_dir, set(checkpoint_paths))

    package = load_graph_package(args.graph_dir, seed=args.seed)

    incremental = _IncrementalReport(
        args.output,
        args.model_dir,
        config={
            "graph_dir": str(args.graph_dir),
            "split_seed": args.seed,
            "model_seeds": model_seeds,
            "epochs": args.epochs,
            "patience": patience,
            "weight_decay": args.weight_decay,
        },
    )
    _run_all(
        package,
        model_seeds,
        epochs=args.epochs,
        weight_decay=args.weight_decay,
        patience=patience,
        incremental=incremental,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
