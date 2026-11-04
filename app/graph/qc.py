"""Graph integrity checks; warnings are audits, errors fail the build."""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from typing import Any

RELATION_TYPES = {
    "IS_A": ("Phenotype", "Phenotype"),
    "HAS_PHENOTYPE": ("Disease", "Phenotype"),
    "ASSOCIATED_WITH": ("Disease", "Gene"),
    "ASSOCIATED_PHENOTYPE": ("Gene", "Phenotype"),
}
WEIGHT_FIELDS = (
    "frequency_weight",
    "evidence_score",
    "idf_raw",
    "idf_norm",
    "idf_penalty_weight",
    "evidence_score_idf",
    "edge_weight",
)


def validate_graph(tables: dict[str, Any]) -> dict[str, Any]:
    nodes, edges = tables["graph_nodes"], tables["graph_edges"]
    errors = Counter(
        {
            key: 0
            for key in (
                "duplicate_node_id",
                "null_node_id",
                "invalid_namespace",
                "source_node_missing",
                "target_node_missing",
                "self_loop",
                "exact_duplicate_edge",
                "canonical_pair_duplicate",
                "invalid_relation",
                "invalid_relation_endpoints",
                "invalid_is_a_reference",
                "invalid_weight",
                "negative_positive_edge",
                "obsolete_hpo_id",
                "ontology_cycle",
                "idf_join_missing",
            )
        }
    )
    by_id: dict[str, dict[str, Any]] = {}
    patterns = {
        "Phenotype": r"HP:[0-9]{7}",
        "Disease": r"ORPHA:[1-9][0-9]*",
        "Gene": r"(?:HGNC:[1-9][0-9]*|ORPHANET_LOCUS:[A-Za-z0-9_.-]+)",
    }
    for node in nodes:
        identifier = node.get("node_id")
        if not identifier:
            errors["null_node_id"] += 1
            continue
        if identifier in by_id:
            errors["duplicate_node_id"] += 1
        by_id[identifier] = node
        if not re.fullmatch(patterns.get(node.get("node_type"), r"(?!)"), identifier):
            errors["invalid_namespace"] += 1
    active_hpo = {row["node_id"] for row in tables.get("hpo_nodes", [])}
    pairs: set[tuple[str, str, str]] = set()
    exact: set[str] = set()
    parents: dict[str, set[str]] = defaultdict(set)
    for edge in edges:
        source, target, relation = (
            edge.get("source_id"),
            edge.get("target_id"),
            edge.get("relation"),
        )
        if source not in by_id:
            errors["source_node_missing"] += 1
        if target not in by_id:
            errors["target_node_missing"] += 1
        if source == target:
            errors["self_loop"] += 1
        pair = (source, relation, target)
        errors["canonical_pair_duplicate"] += pair in pairs
        pairs.add(pair)
        encoded = json.dumps(edge, sort_keys=True, default=str)
        errors["exact_duplicate_edge"] += encoded in exact
        exact.add(encoded)
        if relation not in RELATION_TYPES:
            errors["invalid_relation"] += 1
        elif source in by_id and target in by_id:
            kinds = (by_id[source]["node_type"], by_id[target]["node_type"])
            errors["invalid_relation_endpoints"] += kinds != RELATION_TYPES[relation]
        for field in WEIGHT_FIELDS:
            value = edge.get(field)
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
                or (field != "idf_raw" and value > 1)
            ):
                errors["invalid_weight"] += 1
        if relation in {"HAS_PHENOTYPE", "ASSOCIATED_PHENOTYPE"}:
            errors["negative_positive_edge"] += bool(edge.get("is_negative"))
            errors["obsolete_hpo_id"] += target not in active_hpo
        elif relation == "IS_A":
            errors["obsolete_hpo_id"] += (
                source not in active_hpo or target not in active_hpo
            )
            parents[source].add(target)
        elif relation == "ASSOCIATED_WITH":
            errors["idf_join_missing"] += any(
                edge.get(key) is None
                for key in ("gene_df", "idf_raw", "idf_norm", "evidence_score_idf")
            )
    # Iterative DFS avoids recursion depth assumptions for ontology DAGs.
    state: dict[str, int] = {}
    for start in sorted(parents):
        if state.get(start):
            continue
        stack = [(start, False)]
        while stack:
            node, exiting = stack.pop()
            if exiting:
                state[node] = 2
                continue
            if state.get(node) == 1:
                errors["ontology_cycle"] += 1
                continue
            if state.get(node) == 2:
                continue
            state[node] = 1
            stack.append((node, True))
            stack.extend(
                (parent, False)
                for parent in sorted(parents.get(node, set()), reverse=True)
            )
    errors["invalid_is_a_reference"] = sum(
        row.get("reason") == "invalid_is_a_reference"
        for row in tables.get("unresolved_hpo", [])
    )
    dg = tables.get("disease_gene_edges", [])
    score_ranges = {}
    for field in WEIGHT_FIELDS:
        values = [
            e[field]
            for e in edges
            if isinstance(e.get(field), (int, float)) and math.isfinite(e[field])
        ]
        score_ranges[field] = {
            "min": min(values) if values else None,
            "max": max(values) if values else None,
        }
    return {
        "passed": not any(errors.values()),
        "errors": dict(errors),
        "node_counts": dict(Counter(row["node_type"] for row in nodes)),
        "edge_counts": dict(Counter(row.get("relation") for row in edges)),
        "unresolved_hpo_count": len(tables.get("unresolved_hpo", [])),
        "unresolved_gene_count": len(tables.get("unresolved_genes", [])),
        "gene_canonical_status": dict(
            Counter(row.get("canonical_status") for row in tables.get("gene_nodes", []))
        ),
        "hpo_resolution": dict(
            Counter(
                row.get("hpo_resolution")
                for row in tables.get("mapped_disease_hpo", [])
            )
        ),
        "negative_annotation_count": len(
            tables.get("negative_phenotype_annotations", [])
        ),
        "negative_gene_phenotype_annotation_count": len(
            tables.get("negative_gene_phenotype_annotations", [])
        ),
        "gene_phenotype_unresolved_count": len(
            tables.get("gene_phenotype_unresolved", [])
        ),
        "frequency_distribution": dict(
            Counter(
                str(row.get("frequency"))
                for row in tables.get("cleaned_disease_hpo", [])
            )
        ),
        "association_types": dict(
            Counter(
                row.get("association_type")
                for row in tables.get("cleaned_disease_gene", [])
            )
        ),
        "association_statuses": dict(
            Counter(
                row.get("association_status")
                for row in tables.get("cleaned_disease_gene", [])
            )
        ),
        "disease_hpo_pair_duplicates_removed": sum(
            row["evidence_count"] - 1 for row in tables.get("disease_hpo_edges", [])
        ),
        "disease_gene_pair_duplicates_removed": sum(
            row["evidence_count"] - 1 for row in dg
        ),
        "frequency_conflict_pairs": sum(
            row.get("frequency_conflict", False)
            for row in tables.get("disease_hpo_edges", [])
        ),
        "weight_ranges": score_ranges,
    }
