"""Opt-in canonical graph export, independent of legacy runtime JSON/Neo4j loading."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from app.etl.evidence_scoring import (
    aggregate_disease_gene,
    apply_gene_idf,
    fit_gene_idf,
    score_associations,
)
from app.etl.frequency import FREQUENCY_RULE_VERSION, frequency_weight, is_negative
from app.etl.hpo_loader import load_genes_to_phenotype, load_phenotype_hpoa
from app.etl.id_mapping import HPOCatalog, canonical_disease_id, load_hpo_catalog
from app.etl.orphanet_loader import load_clinical_annotations, load_gene_associations
from app.graph.qc import validate_graph
from app.graph.splits import (
    sample_negative_pairs,
    split_disease_gene,
    training_message_edges,
    validate_split_leakage,
    with_reverse_edges,
)

GRAPH_VERSION = "canonical_graph_v1"


def _json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _edge(source: str, relation: str, target: str, **fields: Any) -> dict[str, Any]:
    return {
        "edge_id": f"{source}|{relation}|{target}",
        "source_id": source,
        "target_id": target,
        "relation": relation,
        **fields,
    }


def _aggregate_hpo(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["disease_id"], row["hpo_id"]].append(row)
    edges = []
    for (disease, hpo), evidence in sorted(groups.items()):
        weights = {
            row["frequency_weight"]
            for row in evidence
            if row["frequency_weight"] is not None
        }
        edges.append(
            _edge(
                disease,
                "HAS_PHENOTYPE",
                hpo,
                frequency_weight=max(weights) if weights else None,
                frequency_conflict=len(weights) > 1,
                frequency=sorted(
                    {row["frequency"] for row in evidence if row["frequency"]}
                ),
                is_negative=False,
                evidence_count=len(evidence),
                evidence_provenance=sorted(evidence, key=_json),
                source_database=sorted({row["source_database"] for row in evidence}),
                frequency_rule_version=FREQUENCY_RULE_VERSION,
            )
        )
    return edges


def _source_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        rows = csv.DictReader(
            (line for line in handle if not line.startswith("#")), delimiter="\t"
        )
        return [
            {
                key.strip().lower().replace("-", "_").replace(" ", "_"): value
                for key, value in row.items()
                if key is not None
            }
            for row in rows
        ]


def _source_value(row: dict[str, Any], *keys: str) -> str:
    # Match the shared loaders' first-present-field and whitespace semantics.
    for key in keys:
        if row.get(key) is not None:
            return row[key].strip()
    return ""


def _hpoa_rows(path: Path) -> list[dict[str, Any]]:
    annotations = load_phenotype_hpoa(path, include_negative=True)
    raw = _source_rows(path)

    def valid(row: dict[str, Any]) -> bool:
        return all(
            (
                _source_value(row, "database_id", "databaseid", "db_object_id"),
                _source_value(
                    row, "disease_name", "diseasename", "db_name", "db_object_name"
                ),
                _source_value(row, "hpo_id"),
            )
        )

    if sum(valid(row) for row in raw) != len(annotations):
        raise ValueError(
            "Canonical HPOA audit alignment differs from team loader; unsupported header"
        )
    accepted = iter(annotations)
    rows = []
    for index, record in enumerate(raw):
        valid_record = valid(record)
        annotation = next(accepted) if valid_record else None
        original_disease = (
            annotation.disease_id
            if annotation
            else _source_value(record, "database_id", "databaseid", "db_object_id")
        )
        frequency = (
            annotation.frequency
            if annotation
            else _source_value(record, "frequency") or None
        )
        rows.append(
            {
                "disease_id": canonical_disease_id(original_disease),
                "disease_id_original": original_disease,
                "disease_name": annotation.disease_name
                if annotation
                else _source_value(
                    record, "disease_name", "diseasename", "db_name", "db_object_name"
                ),
                "hpo_id_original": annotation.hpo_id
                if annotation
                else _source_value(record, "hpo_id"),
                "frequency": frequency,
                "frequency_weight": frequency_weight(frequency),
                "qualifier": _source_value(record, "qualifier"),
                "is_negative": is_negative(
                    _source_value(record, "qualifier"), frequency
                ),
                "source_validation": annotation.evidence
                if annotation
                else _source_value(record, "evidence"),
                "source": annotation.source
                if annotation
                else _source_value(record, "reference", "db_reference"),
                "source_database": "HPOA",
                "source_file": path.name,
                "source_record_id": str(index),
                "raw_annotation": record,
                "ingestion_status": "accepted"
                if valid_record
                else "missing_required_field",
            }
        )
    return rows


def _gene_nodes(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["gene_id"]].append(row)
    nodes = []
    for identifier, records in sorted(groups.items()):
        symbols = sorted({row["gene_symbol"] for row in records if row["gene_symbol"]})
        xrefs: dict[str, set[str]] = defaultdict(set)
        for row in records:
            for namespace, values in row["gene_xrefs"].items():
                xrefs[namespace].update(values)
        nodes.append(
            {
                "node_id": identifier,
                "node_type": "Gene",
                "name": symbols[0] if symbols else identifier,
                "gene_symbol": symbols[0] if symbols else None,
                "gene_symbols_source": symbols,
                "gene_names": sorted(
                    {row["gene_name"] for row in records if row["gene_name"]}
                ),
                "canonical_status": records[0]["canonical_status"],
                "rankable": records[0]["rankable"],
                "source_gene_keys": sorted({row["source_gene_key"] for row in records}),
                "gene_xrefs": {
                    key: sorted(values) for key, values in sorted(xrefs.items())
                },
                "source_database": ["Orphanet"],
                "resolution_basis": "source_declared_hgnc"
                if records[0]["rankable"]
                else "source_locus_fallback",
            }
        )
    return nodes


def _gene_hpo(
    path: Path | None, genes: list[dict[str, Any]], catalog: HPOCatalog
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    if path is None:
        return [], [], [], []
    mapping: dict[str, set[str]] = defaultdict(set)
    for gene in genes:
        if not gene["rankable"]:
            continue
        for namespace, values in gene["gene_xrefs"].items():
            if namespace.lower() in {"entrez", "ncbi", "ncbi gene", "ncbigene"}:
                for value in values:
                    mapping[value].add(gene["node_id"])
    records = _source_rows(path)
    annotations = load_genes_to_phenotype(path)

    def valid(row: dict[str, Any]) -> bool:
        return all(
            (
                _source_value(row, "ncbi_gene_id", "gene_id", "entrez_gene_id"),
                _source_value(row, "gene_symbol", "entrez_gene_symbol", "gene"),
                _source_value(row, "hpo_id", "hpo_term_id"),
            )
        )

    if sum(valid(row) for row in records) != len(annotations):
        raise ValueError("Canonical Gene-HPO audit alignment differs from team loader")
    accepted = iter(annotations)
    raw, unresolved, negatives = [], [], []
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for index, record in enumerate(records):
        valid_record = valid(record)
        parsed = (
            next(accepted).__dict__
            if valid_record
            else {
                "gene_id": _source_value(
                    record, "ncbi_gene_id", "gene_id", "entrez_gene_id"
                ),
                "gene_symbol": _source_value(
                    record, "gene_symbol", "entrez_gene_symbol", "gene"
                ),
                "hpo_id": _source_value(record, "hpo_id", "hpo_term_id"),
                "disease_id": _source_value(record, "disease_id", "disease_id_for_link")
                or None,
            }
        )
        frequency = _source_value(record, "frequency") or None
        annotation = {
            **parsed,
            "raw_annotation": record,
            "frequency": frequency,
            "frequency_weight": frequency_weight(frequency),
            "qualifier": _source_value(record, "qualifier"),
            "is_negative": is_negative(_source_value(record, "qualifier"), frequency),
            "source_file": path.name,
            "source_record_id": str(index),
            "ingestion_status": "accepted"
            if valid_record
            else "missing_required_field",
        }
        raw.append(annotation)
        targets = mapping.get(annotation["gene_id"], set())
        hpo = catalog.resolve(annotation["hpo_id"])
        gene = next(iter(targets)) if len(targets) == 1 else None
        mapped = {
            **annotation,
            "ncbi_gene_id_original": annotation["gene_id"],
            "hpo_id_original": annotation["hpo_id"],
            "canonical_gene_id": gene,
            "canonical_hpo_id": hpo.canonical_id,
            "hpo_resolution": hpo.method,
            "disease_id_original": annotation["disease_id"],
            "mapping_basis": "Orphanet_gene_external_reference" if gene else None,
        }
        if annotation["is_negative"]:
            negatives.append(mapped)
        if not valid_record or gene is None or not hpo.canonical_id:
            reason = (
                "missing_required_field"
                if not valid_record
                else "ncbi_hgnc_mapping_unavailable"
                if not targets
                else "ambiguous_ncbi_hgnc"
                if len(targets) > 1
                else hpo.reason
            )
            unresolved.append({**mapped, "reason": reason})
            continue
        if not annotation["is_negative"]:
            groups[gene, hpo.canonical_id].append(mapped)
    edges = [
        _edge(
            gene,
            "ASSOCIATED_PHENOTYPE",
            hpo,
            evidence_count=len(evidence),
            is_negative=False,
            source_database=["HPO_genes_to_phenotype"],
            evidence_provenance=sorted(evidence, key=_json),
            training_context_eligible=False,
        )
        for (gene, hpo), evidence in sorted(groups.items())
    ]
    return edges, unresolved, raw, negatives


def build_canonical_graph(
    hpo_path: Path,
    hpoa_path: Path,
    clinical_path: Path,
    gene_path: Path,
    genes_to_phenotype_path: Path | None = None,
    *,
    seed: int = 42,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    sources: dict[str, Any] | None = None,
) -> dict[str, Any]:
    catalog = load_hpo_catalog(hpo_path)
    cleaned_hpo = _hpoa_rows(hpoa_path) + load_clinical_annotations(clinical_path)
    cleaned_gene = load_gene_associations(gene_path)
    mapped, positives, negatives, unresolved, non_orpha = [], [], [], [], []
    for row in cleaned_hpo:
        resolution = catalog.resolve(row["hpo_id_original"])
        resolved = {
            **row,
            "hpo_id": resolution.canonical_id,
            "hpo_resolution": resolution.method,
        }
        if row["is_negative"]:
            negatives.append(resolved)
        if resolution.canonical_id is None:
            unresolved.append({**resolved, "reason": resolution.reason})
        if row["disease_id"] is None:
            non_orpha.append(resolved)
            continue
        if row.get("ingestion_status", "accepted") != "accepted":
            continue
        mapped.append(resolved)
        if resolution.canonical_id and not row["is_negative"]:
            positives.append(resolved)
    hpo_nodes = [
        {
            "node_id": term.hpo_id,
            "node_type": "Phenotype",
            "name": term.name,
            "definition": term.definition,
            "synonyms": list(term.synonyms),
            "canonical_status": "resolved",
        }
        for term in sorted(catalog.terms.values(), key=lambda t: t.hpo_id)
    ]
    ontology = []
    for term in sorted(catalog.terms.values(), key=lambda t: t.hpo_id):
        for parent in sorted(set(term.parents)):
            resolution = catalog.resolve(parent)
            if resolution.canonical_id is None:
                unresolved.append(
                    {
                        "hpo_id_original": parent,
                        "source_id": term.hpo_id,
                        "source_file": hpo_path.name,
                        "reason": "invalid_is_a_reference",
                        "hpo_resolution": resolution.method,
                    }
                )
                continue
            ontology.append(
                _edge(
                    term.hpo_id,
                    "IS_A",
                    resolution.canonical_id,
                    source_database=["HPO"],
                    source_version=catalog.release,
                )
            )
    disease_names: dict[str, set[str]] = defaultdict(set)
    for row in [*mapped, *cleaned_gene]:
        disease_names[row["disease_id"]].add(row["disease_name"])
    disease_nodes = [
        {
            "node_id": identifier,
            "node_type": "Disease",
            "name": sorted(names)[0],
            "source_names": sorted(names),
            "canonical_status": "resolved",
            "xrefs": {},
        }
        for identifier, names in sorted(disease_names.items())
    ]
    genes = _gene_nodes(cleaned_gene)
    by_gene = {row["node_id"]: row for row in genes}
    scored = score_associations(cleaned_gene)
    dg = aggregate_disease_gene(scored)
    for row in dg:
        row["edge_id"] = f"{row['source_id']}|ASSOCIATED_WITH|{row['target_id']}"
        row["rankable"] = by_gene[row["target_id"]]["rankable"]
        row["canonical_status"] = by_gene[row["target_id"]]["canonical_status"]
    reference_fit = fit_gene_idf(dg, scope="full_reference")
    dg = apply_gene_idf(dg, reference_fit)
    dh = _aggregate_hpo(positives)
    gh, gene_unresolved, gene_raw, gene_negatives = _gene_hpo(
        genes_to_phenotype_path, genes, catalog
    )
    splits = split_disease_gene(
        [row for row in dg if row["rankable"]],
        seed=seed,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
    )
    train_fit = fit_gene_idf(splits["train"], scope="train_only")
    splits = {key: apply_gene_idf(rows, train_fit) for key, rows in splits.items()}
    full_pairs = {(row["source_id"], row["target_id"]) for row in dg}
    negative_count = sum(len(rows) for rows in splits.values())
    negative_capacity = len(disease_nodes) * sum(
        row["rankable"] for row in genes
    ) - sum(target.startswith("HGNC:") for _, target in full_pairs)
    negative_pairs = sample_negative_pairs(
        [row["node_id"] for row in disease_nodes],
        [row["node_id"] for row in genes if row["rankable"]],
        full_pairs,
        count=min(negative_count, negative_capacity),
        seed=seed,
    )
    split_reverse = {key: with_reverse_edges(rows) for key, rows in splits.items()}
    split_qc = validate_split_leakage(
        split_reverse,
        known_positive_pairs=full_pairs,
        negative_pairs=negative_pairs,
        idf_fit=train_fit,
    )
    tables: dict[str, Any] = {
        "hpo_nodes": hpo_nodes,
        "disease_nodes": disease_nodes,
        "gene_nodes": genes,
        "hpo_hpo_edges": ontology,
        "disease_hpo_edges": dh,
        "disease_gene_edges": dg,
        "gene_hpo_edges": gh,
        "graph_nodes": [*hpo_nodes, *disease_nodes, *genes],
        "graph_edges": [*ontology, *dh, *dg, *gh],
        "cleaned_disease_hpo": cleaned_hpo,
        "mapped_disease_hpo": mapped,
        "cleaned_disease_gene": scored,
        "cleaned_gene_phenotype": gene_raw,
        "unresolved_hpo": unresolved,
        "negative_phenotype_annotations": negatives,
        "non_orpha_annotations": non_orpha,
        "rejected_source_annotations": [
            row
            for row in cleaned_hpo
            if row.get("ingestion_status", "accepted") != "accepted"
        ],
        "unresolved_genes": [row for row in genes if not row["rankable"]],
        "gene_phenotype_unresolved": gene_unresolved,
        "negative_gene_phenotype_annotations": gene_negatives,
        "splits": splits,
        "split_reverse": split_reverse,
        "negative_pairs": [
            {"source_id": d, "target_id": g, "label": 0} for d, g in negative_pairs
        ],
        "idf_fit": train_fit,
        "idf_reference_fit": reference_fit,
        "split_qc": split_qc,
        "training_message_edges": training_message_edges([*ontology, *dh], splits),
    }
    tables["qc_report"] = validate_graph(tables)
    tables["qc_report"]["split"] = split_qc
    tables["qc_report"]["passed"] &= split_qc["passed"]
    tables["graph_metadata"] = {
        "graph_version": GRAPH_VERSION,
        "hpo_release": catalog.release,
        "node_count": len(tables["graph_nodes"]),
        "edge_count": len(tables["graph_edges"]),
        "relation_counts": tables["qc_report"]["edge_counts"],
        "node_counts": tables["qc_report"]["node_counts"],
        "source_manifest": sources,
        "frequency_rule_version": FREQUENCY_RULE_VERSION,
        "disease_universe": "union",
        "disease_hpo_aggregation": "max_known_frequency;NULL_if_all_missing",
        "split": {
            "seed": seed,
            "validation_fraction": val_fraction,
            "test_fraction": test_fraction,
            "version": "dg_link_v1",
            "task": "disease_gene_link_prediction",
            "sizes": {key: len(rows) for key, rows in splits.items()},
            "negative_requested": negative_count,
            "negative_generated": len(negative_pairs),
            "negative_capacity": negative_capacity,
            "negative_scope": "all_known_positive_exclusion",
            "nonrankable_target_edges_excluded": sum(not row["rankable"] for row in dg),
            "gene_hpo_training_context": "quarantined_all",
            "disease_hpo_role": "input_context_not_target",
        },
        "runtime_compatibility": "opt_in_export;legacy_JSON_and_Neo4j_unchanged",
    }
    return tables


NODE_FIELDS = ["node_id", "node_type", "name", "canonical_status"]
EDGE_FIELDS = ["source_id", "target_id", "relation", "edge_id", "source_database"]
DG_FIELDS = [
    *EDGE_FIELDS,
    "evidence_count",
    "evidence_provenance",
    "evidence_score",
    "gene_df",
    "idf_raw",
    "idf_norm",
    "idf_penalty_weight",
    "evidence_score_idf",
    "idf_scope",
    "rankable",
    "canonical_status",
    "association_type",
    "association_status",
    "source_validation",
    "weight_rule_version",
]
HPO_AUDIT_FIELDS = [
    "source_record_id",
    "source_file",
    "disease_id",
    "disease_id_original",
    "disease_name",
    "hpo_id_original",
    "hpo_id",
    "frequency",
    "frequency_weight",
    "qualifier",
    "is_negative",
    "raw_annotation",
    "hpo_resolution",
    "ingestion_status",
]
GENE_HPO_AUDIT_FIELDS = [
    "source_record_id",
    "source_file",
    "gene_id",
    "gene_symbol",
    "hpo_id",
    "disease_id",
    "frequency",
    "frequency_weight",
    "qualifier",
    "is_negative",
    "raw_annotation",
    "ingestion_status",
    "canonical_gene_id",
    "canonical_hpo_id",
    "hpo_resolution",
    "mapping_basis",
]
TABLE_FIELDS = {
    "hpo_nodes": [*NODE_FIELDS, "definition", "synonyms"],
    "disease_nodes": [*NODE_FIELDS, "source_names", "xrefs"],
    "gene_nodes": [
        *NODE_FIELDS,
        "gene_symbol",
        "gene_symbols_source",
        "gene_names",
        "rankable",
        "source_gene_keys",
        "gene_xrefs",
        "source_database",
        "resolution_basis",
    ],
    "hpo_hpo_edges": [*EDGE_FIELDS, "source_version"],
    "disease_hpo_edges": [
        *EDGE_FIELDS,
        "frequency_weight",
        "frequency_conflict",
        "frequency",
        "is_negative",
        "evidence_count",
        "evidence_provenance",
        "frequency_rule_version",
    ],
    "disease_gene_edges": DG_FIELDS,
    "gene_hpo_edges": [
        *EDGE_FIELDS,
        "evidence_count",
        "evidence_provenance",
        "is_negative",
        "training_context_eligible",
    ],
    "cleaned_disease_hpo": HPO_AUDIT_FIELDS,
    "mapped_disease_hpo": HPO_AUDIT_FIELDS,
    "negative_phenotype_annotations": HPO_AUDIT_FIELDS,
    "non_orpha_annotations": HPO_AUDIT_FIELDS,
    "rejected_source_annotations": HPO_AUDIT_FIELDS,
    "unresolved_hpo": [*HPO_AUDIT_FIELDS, "reason", "source_id"],
    "cleaned_gene_phenotype": GENE_HPO_AUDIT_FIELDS,
    "gene_phenotype_unresolved": [*GENE_HPO_AUDIT_FIELDS, "reason"],
    "negative_gene_phenotype_annotations": GENE_HPO_AUDIT_FIELDS,
    "negative_pairs": ["source_id", "target_id", "label"],
    "training_message_edges": DG_FIELDS,
}
TABLE_FIELDS["unresolved_genes"] = TABLE_FIELDS["gene_nodes"]


def _write_csv(
    path: Path, rows: list[dict[str, Any]], default_fields: list[str]
) -> None:
    fields = [
        *default_fields,
        *sorted({key for row in rows for key in row} - set(default_fields)),
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: _json(value)
                    if isinstance(value, (dict, list, bool))
                    else value
                    for key, value in row.items()
                }
            )


def write_graph(tables: dict[str, Any], output_dir: Path) -> None:
    """Write relation-specific source-of-truth tables and deterministic unified views."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in tables.items():
        if not isinstance(rows, list):
            continue
        fields = (
            ["node_id", "node_type", "name"]
            if name.endswith("nodes")
            else ["source_id", "target_id", "relation"]
            if name.endswith("edges")
            else ["source_record_id"]
        )
        _write_csv(output_dir / f"{name}.csv", rows, TABLE_FIELDS.get(name, fields))
    split_dir = output_dir / "splits"
    split_dir.mkdir(exist_ok=True)
    for name, rows in tables["splits"].items():
        _write_csv(
            split_dir / f"{name}.csv",
            rows,
            [*DG_FIELDS, "split", "seed", "split_version"],
        )
        _write_csv(
            split_dir / f"{name}_with_reverse.csv",
            tables["split_reverse"][name],
            [*DG_FIELDS, "split", "seed", "split_version", "reverse_of"],
        )
    for name in ("idf_fit", "idf_reference_fit", "split_qc"):
        (split_dir / f"{name}.json").write_text(
            json.dumps(
                tables[name],
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
    for name in ("graph_metadata", "qc_report"):
        (output_dir / f"{name}.json").write_text(
            json.dumps(
                tables[name],
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
    ablations = output_dir / "ablations"
    ablations.mkdir(exist_ok=True)
    for mode, field in (
        ("A", "evidence_score"),
        ("B", "evidence_score_idf"),
        ("C", None),
    ):
        rows = [
            {
                **edge,
                "ablation": mode,
                **({"edge_weight": edge[field]} if field else {}),
            }
            for edge in tables["splits"]["train"]
        ]
        _write_csv(
            ablations / f"{mode}_train.csv",
            rows,
            [
                *DG_FIELDS,
                "split",
                "seed",
                "split_version",
                "ablation",
                *(["edge_weight"] if field else []),
            ],
        )
    # Hashes omit their own container so repeated builds are byte-identical.
    hashes = {
        str(path.relative_to(output_dir)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(output_dir.rglob("*"))
        if path.is_file() and path.name != "artifact_hashes.json"
    }
    (output_dir / "artifact_hashes.json").write_text(
        json.dumps(hashes, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
