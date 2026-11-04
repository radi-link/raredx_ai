"""Deterministic disease-gene splits with explicit leakage boundaries."""

from __future__ import annotations

import copy
import math
import random
import re
from collections import Counter
from typing import Any, Iterable

from app.etl.evidence_scoring import fit_gene_idf

SPLIT_VERSION = "dg_link_v1"
SPLIT_NAMES = ("train", "validation", "test")
Edge = dict[str, Any]
Pair = tuple[str, str]


def _identifier(value: Any, namespace: str) -> bool:
    return (
        isinstance(value, str)
        and re.fullmatch(rf"{namespace}:[1-9][0-9]*", value) is not None
    )


def _pair(edge: Edge, *, reverse_allowed: bool = False) -> Pair:
    relation = edge.get("relation")
    if relation == "ASSOCIATED_WITH_REVERSE":
        if not reverse_allowed:
            raise ValueError("expected forward ASSOCIATED_WITH edges")
        disease, gene = edge.get("target_id"), edge.get("source_id")
    elif relation == "ASSOCIATED_WITH":
        disease, gene = edge.get("source_id"), edge.get("target_id")
    else:
        raise ValueError("expected ASSOCIATED_WITH relation")
    if not _identifier(disease, "ORPHA") or not _identifier(gene, "HGNC"):
        raise ValueError("rankable targets require numeric ORPHA and HGNC identifiers")
    if (
        edge.get("rankable", True) is not True
        or edge.get("unresolved", False) is not False
    ):
        raise ValueError("non-rankable or unresolved prediction target")
    return disease, gene


def split_disease_gene(
    edges: Iterable[Edge],
    *,
    seed: int = 42,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
) -> dict[str, list[Edge]]:
    """Split canonical forward targets; reference-only rows must be filtered by caller."""
    for fraction in (val_fraction, test_fraction):
        if (
            isinstance(fraction, bool)
            or not isinstance(fraction, (int, float))
            or not math.isfinite(fraction)
            or fraction < 0
        ):
            raise ValueError("fractions must be finite and nonnegative")
    if val_fraction + test_fraction >= 1:
        raise ValueError("holdout fractions must sum to less than one")
    keyed: dict[Pair, Edge] = {}
    for edge in edges:
        pair = _pair(edge)
        if pair in keyed:
            raise ValueError("duplicate disease-gene pair")
        keyed[pair] = edge
    pairs = sorted(keyed)
    random.Random(seed).shuffle(pairs)
    n_val, n_test = int(len(pairs) * val_fraction), int(len(pairs) * test_fraction)
    groups = {
        "validation": pairs[:n_val],
        "test": pairs[n_val : n_val + n_test],
        "train": pairs[n_val + n_test :],
    }
    return {
        name: [
            {
                **copy.deepcopy(keyed[pair]),
                "split": name,
                "seed": seed,
                "split_version": SPLIT_VERSION,
            }
            for pair in groups[name]
        ]
        for name in SPLIT_NAMES
    }


def with_reverse_edges(edges: Iterable[Edge]) -> list[Edge]:
    result: list[Edge] = []
    seen: set[Pair] = set()
    for edge in edges:
        pair = _pair(edge)
        if pair in seen:
            raise ValueError("duplicate disease-gene pair")
        seen.add(pair)
        forward = copy.deepcopy(edge)
        identity = edge.get("edge_id") or f"{pair[0]}|ASSOCIATED_WITH|{pair[1]}"
        result.extend(
            [
                forward,
                {
                    **copy.deepcopy(forward),
                    "source_id": pair[1],
                    "target_id": pair[0],
                    "relation": "ASSOCIATED_WITH_REVERSE",
                    "edge_id": f"{identity}:reverse",
                    "reverse_of": identity,
                },
            ]
        )
    return result


def sample_negative_pairs(
    disease_ids: Iterable[str],
    gene_ids: Iterable[str],
    known_positive_pairs: Iterable[Pair],
    *,
    count: int,
    seed: int = 42,
) -> list[Pair]:
    """Sample uniformly without replacement using compressed eligible indices."""
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError("count must be a nonnegative integer")
    diseases, genes = sorted(set(disease_ids)), sorted(set(gene_ids))
    if any(not _identifier(v, "ORPHA") for v in diseases) or any(
        not _identifier(v, "HGNC") for v in genes
    ):
        raise ValueError(
            "rankable negative targets require numeric ORPHA and HGNC identifiers"
        )
    disease_index = {value: index for index, value in enumerate(diseases)}
    gene_index = {value: index for index, value in enumerate(genes)}
    blocked = sorted(
        {
            disease_index[d] * len(genes) + gene_index[g]
            for d, g in known_positive_pairs
            if d in disease_index and g in gene_index
        }
    )
    capacity = len(diseases) * len(genes) - len(blocked)
    if count > capacity:
        raise ValueError("count exceeds available negative capacity")
    # Each blocked index removes one position. Binary search maps an eligible
    # rank back to its Cartesian index without allocating the Cartesian product.
    from bisect import bisect_right

    result: list[Pair] = []
    for rank in random.Random(seed).sample(range(capacity), count):
        low, high = rank, rank + len(blocked)
        while low < high:
            mid = (low + high) // 2
            if mid - bisect_right(blocked, mid) < rank:
                low = mid + 1
            else:
                high = mid
        result.append((diseases[low // len(genes)], genes[low % len(genes)]))
    return result


def validate_split_leakage(
    splits: dict[str, list[Edge]],
    *,
    known_positive_pairs: Iterable[Pair] | None = None,
    negative_pairs: Iterable[Pair] | None = None,
    idf_fit: dict[str, Any] | None = None,
) -> dict[str, Any]:
    counts = {
        name: 0
        for name in (
            "pair_overlap_across_splits",
            "reverse_split_conflicts",
            "duplicate_pairs_within_split",
            "known_positive_negative_collisions",
            "duplicate_negative_pairs",
            "invalid_edges",
            "idf_fit_metadata_errors",
        )
    }
    locations: dict[Pair, set[str]] = {}
    forwards: dict[Pair, list[tuple[str, Edge]]] = {}
    reverses: list[tuple[Pair, str, Edge]] = []
    train: list[Edge] = []
    for name, rows in splits.items():
        seen: set[tuple[Pair, str]] = set()
        for edge in rows:
            try:
                pair = _pair(edge, reverse_allowed=True)
            except ValueError:
                counts["invalid_edges"] += 1
                continue
            relation = edge["relation"]
            key = (pair, relation)
            if key in seen:
                counts["duplicate_pairs_within_split"] += 1
            seen.add(key)
            locations.setdefault(pair, set()).add(name)
            if name not in SPLIT_NAMES or edge.get("split", name) != name:
                counts["reverse_split_conflicts"] += 1
            if relation == "ASSOCIATED_WITH":
                forwards.setdefault(pair, []).append((name, edge))
                if name == "train":
                    train.append(edge)
            else:
                reverses.append((pair, name, edge))
    counts["pair_overlap_across_splits"] = sum(
        len(names) > 1 for names in locations.values()
    )
    for pair, name, edge in reverses:
        matches = forwards.get(pair, [])
        if not matches or not any(
            split == name
            and edge.get("reverse_of")
            == (forward.get("edge_id") or f"{pair[0]}|ASSOCIATED_WITH|{pair[1]}")
            and all(
                edge.get(field) == forward.get(field)
                for field in ("seed", "split_version")
            )
            for split, forward in matches
        ):
            counts["reverse_split_conflicts"] += 1
    negatives = list(negative_pairs or [])
    positives = set(known_positive_pairs or []) | set(locations)
    counts["known_positive_negative_collisions"] = sum(
        pair in positives for pair in negatives
    )
    counts["duplicate_negative_pairs"] = sum(n - 1 for n in Counter(negatives).values())
    counts["invalid_edges"] += sum(
        not _identifier(d, "ORPHA") or not _identifier(g, "HGNC") for d, g in negatives
    )
    if idf_fit is not None:
        expected = fit_gene_idf(train, scope="train_only")
        for key, value in expected.items():
            if idf_fit.get(key) != value:
                counts["idf_fit_metadata_errors"] += 1
    return {"passed": not any(counts.values()), "counts": counts}


def training_message_edges(
    structural_edges: Iterable[Edge],
    splits: dict[str, list[Edge]],
    *,
    prediction_target: str = "disease_gene",
) -> list[Edge]:
    """Construct DG messages; all G-HPO provenance stays in the reference graph."""
    if prediction_target != "disease_gene":
        raise ValueError(
            "unsafe prediction_target: phenotype targets require a separate leakage policy"
        )
    structural: list[Edge] = []
    for edge in structural_edges:
        relation = edge.get("relation")
        if relation == "ASSOCIATED_PHENOTYPE":
            continue
        if relation not in {"IS_A", "HAS_PHENOTYPE"}:
            raise ValueError("unsupported structural relation")
        structural.append(copy.deepcopy(edge))
    report = validate_split_leakage(splits)
    if not report["passed"]:
        raise ValueError(f"unsafe disease-gene splits: {report['counts']}")
    return structural + copy.deepcopy(splits.get("train", []))
