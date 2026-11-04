"""Deterministic disease-gene evidence scoring and IDF transforms."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections import defaultdict
from typing import Any

WEIGHT_RULE_VERSION = "v1.0"
IDF_RULE_VERSION = "idf_v1.0"
IDF_MIN_WEIGHT = 0.5

ASSOCIATION_TYPE_WEIGHTS: dict[str, float] = {
    "Disease-causing germline mutation(s) in": 1.0,
    "Disease-causing germline mutation(s) (loss of function) in": 1.0,
    "Disease-causing germline mutation(s) (gain of function) in": 1.0,
    "Disease-causing somatic mutation(s) in": 0.8,
    "Major susceptibility factor in": 0.6,
    "Modifying germline mutation in": 0.5,
    "Part of a fusion gene in": 0.5,
    "Role in the phenotype of": 0.4,
    "Candidate gene tested in": 0.3,
    "Biomarker tested in": 0.2,
}

ASSOCIATION_STATUS_WEIGHTS: dict[str, float] = {
    "Assessed": 1.0,
    "Not yet assessed": 0.5,
    "Not assessed": 0.5,
    "missing": 0.5,
}

IDF_SCOPES = frozenset({"full_reference", "train_only"})


def score_associations(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Copy and score raw Orphanet disease-gene association rows."""
    scored_rows: list[dict[str, Any]] = []
    for row in rows:
        association_type = row.get("association_type")
        if association_type not in ASSOCIATION_TYPE_WEIGHTS:
            raise ValueError(f"unknown association_type: {association_type!r}")

        raw_status = row.get("association_status")
        association_status = (
            "missing" if raw_status is None or raw_status == "" else raw_status
        )
        if association_status not in ASSOCIATION_STATUS_WEIGHTS:
            raise ValueError(f"unknown association_status: {association_status!r}")

        association_type_weight = ASSOCIATION_TYPE_WEIGHTS[association_type]
        association_status_weight = ASSOCIATION_STATUS_WEIGHTS[association_status]
        scored = copy.deepcopy(row)
        scored.update(
            {
                "association_status": association_status,
                "association_type_weight": association_type_weight,
                "association_status_weight": association_status_weight,
                "evidence_score": association_type_weight * association_status_weight,
                "weight_rule_version": WEIGHT_RULE_VERSION,
            }
        )
        scored_rows.append(scored)
    return scored_rows


def aggregate_disease_gene(scored_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate scored evidence into one canonical edge per disease-gene pair."""
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in scored_rows:
        disease_id = _required_id(row, "disease_id")
        gene_id = _required_id(row, "gene_id")
        grouped[(disease_id, gene_id)].append(copy.deepcopy(row))

    edges: list[dict[str, Any]] = []
    for (disease_id, gene_id), evidence_rows in sorted(grouped.items()):
        evidence_provenance = sorted(evidence_rows, key=_canonical_json)
        scores = [
            _finite_number(row.get("evidence_score"), "evidence_score")
            for row in evidence_rows
        ]
        edges.append(
            {
                "source_id": disease_id,
                "target_id": gene_id,
                "relation": "ASSOCIATED_WITH",
                "association_type": _sorted_strings(evidence_rows, "association_type"),
                "association_status": _sorted_strings(
                    evidence_rows, "association_status"
                ),
                "evidence_score": max(scores),
                "evidence_count": len(evidence_rows),
                "source_validation": _sorted_strings(
                    evidence_rows, "source_validation"
                ),
                "source_database": _sorted_strings(evidence_rows, "source_database"),
                "weight_rule_version": WEIGHT_RULE_VERSION,
                "evidence_provenance": evidence_provenance,
            }
        )
    return edges


def fit_gene_idf(edges: list[dict[str, Any]], *, scope: str) -> dict[str, Any]:
    """Fit auditable IDF statistics from canonical disease-gene pairs only."""
    if scope not in IDF_SCOPES:
        raise ValueError(f"unknown IDF scope: {scope!r}")

    unique_pairs = sorted(
        {
            (_required_id(edge, "source_id"), _required_id(edge, "target_id"))
            for edge in edges
        }
    )
    disease_count_total = len({source_id for source_id, _ in unique_pairs})
    genes_to_diseases: dict[str, set[str]] = defaultdict(set)
    for source_id, target_id in unique_pairs:
        genes_to_diseases[target_id].add(source_id)

    gene_df = {
        target_id: len(genes_to_diseases[target_id])
        for target_id in sorted(genes_to_diseases)
    }
    idf_values = [
        math.log((disease_count_total + 1) / (document_frequency + 1))
        for document_frequency in gene_df.values()
    ]
    idf_min = min(idf_values) if idf_values else 0.0
    idf_max = max(idf_values) if idf_values else 0.0
    pairs = [[source_id, target_id] for source_id, target_id in unique_pairs]
    pair_payload = json.dumps(pairs, ensure_ascii=False, separators=(",", ":"))

    return {
        "pairs": pairs,
        "gene_df": gene_df,
        "idf_min": idf_min,
        "idf_max": idf_max,
        "disease_count_total": disease_count_total,
        "idf_scope": scope,
        "idf_rule_version": IDF_RULE_VERSION,
        "fit_pair_sha256": hashlib.sha256(pair_payload.encode("utf-8")).hexdigest(),
    }


def apply_gene_idf(
    edges: list[dict[str, Any]],
    fit: dict[str, Any],
) -> list[dict[str, Any]]:
    """Apply frozen IDF statistics to edges without refitting on held-out data."""
    scope = fit.get("idf_scope")
    if scope not in IDF_SCOPES:
        raise ValueError(f"unknown IDF scope: {scope!r}")

    disease_count_total = int(fit["disease_count_total"])
    gene_df = fit["gene_df"]
    idf_min = _finite_number(fit["idf_min"], "idf_min")
    idf_max = _finite_number(fit["idf_max"], "idf_max")

    transformed: list[dict[str, Any]] = []
    for edge in edges:
        target_id = _required_id(edge, "target_id")
        document_frequency = int(gene_df.get(target_id, 0))
        idf_raw = math.log((disease_count_total + 1) / (document_frequency + 1))
        if idf_max == idf_min:
            idf_norm = 1.0
        else:
            idf_norm = (idf_raw - idf_min) / (idf_max - idf_min)
            idf_norm = min(1.0, max(0.0, idf_norm))
        idf_penalty_weight = IDF_MIN_WEIGHT + (1.0 - IDF_MIN_WEIGHT) * idf_norm
        evidence_score = _finite_number(edge.get("evidence_score"), "evidence_score")

        scored_edge = copy.deepcopy(edge)
        scored_edge.update(
            {
                "gene_df": document_frequency,
                "idf_raw": idf_raw,
                "idf_norm": idf_norm,
                "idf_min_weight": IDF_MIN_WEIGHT,
                "idf_penalty_weight": idf_penalty_weight,
                "evidence_score_idf": evidence_score * idf_penalty_weight,
                "idf_scope": scope,
                "idf_rule_version": IDF_RULE_VERSION,
                "disease_count_total": disease_count_total,
            }
        )
        transformed.append(scored_edge)
    return transformed


def _required_id(row: dict[str, Any], field: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field} must be a finite number")
    return number


def _sorted_strings(rows: list[dict[str, Any]], field: str) -> list[str]:
    return sorted(
        {value for row in rows if isinstance((value := row.get(field)), str) and value}
    )


def _canonical_json(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
