"""CLI: train baseline and GNN disease-gene link predictors, write a JSON report.

Trains `BaselineScorer` and `GNNLinkPredictor` (see `app/modeling/train.py`) on a
graph package (see `app/modeling/loader.py`), evaluates each on the validation and
test splits via `evaluate_link_prediction` (see `app/modeling/evaluate.py`), and
writes a single JSON report with top-level keys `"baseline"` and `"gnn"`, each
holding `{"validation": ..., "test": ...}` serialized `EvaluationReport`s.

Also saves each trained model's `state_dict` under `--model-dir` so the run can be
shared and reloaded without retraining. Refuses to overwrite an existing checkpoint
or report file unless `--overwrite` is passed.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path
from typing import Callable

sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch

from app.modeling.evaluate import (
    DEFAULT_CANDIDATE_POOL_SIZE,
    DEFAULT_NUM_DISEASES,
    CandidateSubsetMetrics,
    EvaluationReport,
    MultiDiseaseCandidateMetrics,
    evaluate_candidate_subset_ranking,
    evaluate_link_prediction,
    evaluate_multi_disease_candidate_pool,
)
from app.modeling.loader import GraphPackage, load_graph_package
from app.modeling.models import BaselineScorer, GNNLinkPredictor
from app.modeling.train import train_baseline, train_gnn

Pair = tuple[str, str]
ScorePairFn = Callable[[list[Pair]], list[float]]


def _baseline_score_pair(model: BaselineScorer, package: GraphPackage) -> ScorePairFn:
    """Adapt `BaselineScorer` to the `score_pair` callable contract."""
    disease_index = package.node_indexes["Disease"].index
    gene_index = package.node_indexes["Gene"].index

    model.eval()

    def score_pair(pairs: list[Pair]) -> list[float]:
        disease_idx = torch.tensor(
            [disease_index[disease_id] for disease_id, _gene_id in pairs], dtype=torch.long
        )
        gene_idx = torch.tensor(
            [gene_index[gene_id] for _disease_id, gene_id in pairs], dtype=torch.long
        )
        with torch.no_grad():
            logits = model(disease_idx, gene_idx)
        return logits.tolist()

    return score_pair


def _gnn_score_pair(model: GNNLinkPredictor, package: GraphPackage) -> ScorePairFn:
    """Adapt `GNNLinkPredictor` to the `score_pair` callable contract.

    Encodes the graph once up front (eval mode, no grad) and caches the resulting
    Disease/Gene embeddings, so repeated `score_pair` calls during evaluation only
    run the cheap bilinear decode, not the full GNN forward pass.
    """
    disease_index = package.node_indexes["Disease"].index
    gene_index = package.node_indexes["Gene"].index

    model.eval()
    with torch.no_grad():
        embeddings = model.encode(package.data)
    disease_emb = embeddings["Disease"]
    gene_emb = embeddings["Gene"]

    def score_pair(pairs: list[Pair]) -> list[float]:
        disease_idx = torch.tensor(
            [disease_index[disease_id] for disease_id, _gene_id in pairs], dtype=torch.long
        )
        gene_idx = torch.tensor(
            [gene_index[gene_id] for _disease_id, gene_id in pairs], dtype=torch.long
        )
        with torch.no_grad():
            logits = model.decode(disease_emb, gene_emb, disease_idx, gene_idx)
        return logits.tolist()

    return score_pair


def _report_to_dict(report: EvaluationReport) -> dict:
    return dataclasses.asdict(report)


def _candidate_subset_to_dict(metrics: CandidateSubsetMetrics) -> dict:
    return dataclasses.asdict(metrics)


def _multi_disease_candidate_pool_to_dict(metrics: MultiDiseaseCandidateMetrics) -> dict:
    return dataclasses.asdict(metrics)


def _check_no_clobber(paths: list[Path], *, overwrite: bool) -> None:
    if overwrite:
        return
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise FileExistsError(
            "refusing to overwrite existing output file(s) without --overwrite: "
            + ", ".join(existing)
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Train baseline and GNN disease-gene link prediction models and write "
            "a filtered, bidirectional Hits@K/MRR evaluation report."
        )
    )
    parser.add_argument("--graph-dir", type=Path, default=Path("data/processed/graph"))
    parser.add_argument("--seed", type=int, default=42)
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
        "--candidate-pool-size",
        type=int,
        default=DEFAULT_CANDIDATE_POOL_SIZE,
        help=(
            "size of the disease-specific gene candidate pool used by the "
            "candidate_subset report section (models ranking a small, "
            "already-connected gene set per disease rather than the full "
            "gene universe)"
        ),
    )
    parser.add_argument(
        "--num-candidate-diseases",
        type=int,
        default=DEFAULT_NUM_DISEASES,
        help=(
            "number of 'Top-N' predicted diseases whose known genes are unioned "
            "into the multi_disease_candidate_pool report section's candidate "
            "pool (models the service's disease TOP-N -> gene TOP-K pipeline; "
            "since this package carries no disease-ranking signal, the "
            "(N - 1) non-true diseases are a random stand-in, not a real "
            "prediction -- see evaluate_multi_disease_candidate_pool)"
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="allow overwriting an existing report or checkpoint at the output paths",
    )
    args = parser.parse_args(argv)
    patience = None if args.patience <= 0 else args.patience

    baseline_checkpoint = args.model_dir / "baseline_state_dict.pt"
    gnn_checkpoint = args.model_dir / "gnn_state_dict.pt"
    _check_no_clobber(
        [args.output, baseline_checkpoint, gnn_checkpoint], overwrite=args.overwrite
    )

    package = load_graph_package(args.graph_dir, seed=args.seed)

    baseline_model, baseline_history = train_baseline(
        package,
        epochs=args.epochs,
        seed=args.seed,
        patience=patience,
        weight_decay=args.weight_decay,
    )
    gnn_model, gnn_history = train_gnn(
        package,
        epochs=args.epochs,
        seed=args.seed,
        patience=patience,
        weight_decay=args.weight_decay,
    )

    baseline_score_pair = _baseline_score_pair(baseline_model, package)
    gnn_score_pair = _gnn_score_pair(gnn_model, package)

    report = {
        "training": {
            "baseline": {
                "epochs_run": len(baseline_history.train_loss),
                "best_epoch": baseline_history.best_epoch,
                "best_validation_loss": min(baseline_history.validation_loss),
            },
            "gnn": {
                "epochs_run": len(gnn_history.train_loss),
                "best_epoch": gnn_history.best_epoch,
                "best_validation_loss": min(gnn_history.validation_loss),
            },
        },
        "baseline": {
            "validation": _report_to_dict(
                evaluate_link_prediction(baseline_score_pair, package, "validation")
            ),
            "test": _report_to_dict(
                evaluate_link_prediction(baseline_score_pair, package, "test")
            ),
        },
        "gnn": {
            "validation": _report_to_dict(
                evaluate_link_prediction(gnn_score_pair, package, "validation")
            ),
            "test": _report_to_dict(evaluate_link_prediction(gnn_score_pair, package, "test")),
        },
        "candidate_subset": {
            "baseline": {
                "validation": _candidate_subset_to_dict(
                    evaluate_candidate_subset_ranking(
                        baseline_score_pair,
                        package,
                        "validation",
                        pool_size=args.candidate_pool_size,
                        seed=args.seed,
                    )
                ),
                "test": _candidate_subset_to_dict(
                    evaluate_candidate_subset_ranking(
                        baseline_score_pair,
                        package,
                        "test",
                        pool_size=args.candidate_pool_size,
                        seed=args.seed,
                    )
                ),
            },
            "gnn": {
                "validation": _candidate_subset_to_dict(
                    evaluate_candidate_subset_ranking(
                        gnn_score_pair,
                        package,
                        "validation",
                        pool_size=args.candidate_pool_size,
                        seed=args.seed,
                    )
                ),
                "test": _candidate_subset_to_dict(
                    evaluate_candidate_subset_ranking(
                        gnn_score_pair,
                        package,
                        "test",
                        pool_size=args.candidate_pool_size,
                        seed=args.seed,
                    )
                ),
            },
        },
        "multi_disease_candidate_pool": {
            "baseline": {
                "validation": _multi_disease_candidate_pool_to_dict(
                    evaluate_multi_disease_candidate_pool(
                        baseline_score_pair,
                        package,
                        "validation",
                        num_diseases=args.num_candidate_diseases,
                        pool_size=args.candidate_pool_size,
                        seed=args.seed,
                    )
                ),
                "test": _multi_disease_candidate_pool_to_dict(
                    evaluate_multi_disease_candidate_pool(
                        baseline_score_pair,
                        package,
                        "test",
                        num_diseases=args.num_candidate_diseases,
                        pool_size=args.candidate_pool_size,
                        seed=args.seed,
                    )
                ),
            },
            "gnn": {
                "validation": _multi_disease_candidate_pool_to_dict(
                    evaluate_multi_disease_candidate_pool(
                        gnn_score_pair,
                        package,
                        "validation",
                        num_diseases=args.num_candidate_diseases,
                        pool_size=args.candidate_pool_size,
                        seed=args.seed,
                    )
                ),
                "test": _multi_disease_candidate_pool_to_dict(
                    evaluate_multi_disease_candidate_pool(
                        gnn_score_pair,
                        package,
                        "test",
                        num_diseases=args.num_candidate_diseases,
                        pool_size=args.candidate_pool_size,
                        seed=args.seed,
                    )
                ),
            },
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    args.model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(baseline_model.state_dict(), baseline_checkpoint)
    torch.save(gnn_model.state_dict(), gnn_checkpoint)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
