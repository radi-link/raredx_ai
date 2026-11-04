"""Raw Orphanet associations, preserving negatives and source-declared gene xrefs."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from app.etl.frequency import frequency_weight, is_negative
from app.etl.id_mapping import canonical_disease_id, canonical_gene


def _tag(element: ET.Element) -> str:
    return element.tag.rsplit("}", 1)[-1]


def _children(element: ET.Element | None, tag: str) -> list[ET.Element]:
    return [] if element is None else [child for child in element if _tag(child) == tag]


def _child(element: ET.Element | None, tag: str) -> ET.Element | None:
    matches = _children(element, tag)
    return matches[0] if matches else None


def _text(element: ET.Element | None, *path: str) -> str:
    for tag in path:
        element = _child(element, tag)
    return " ".join((element.text or "").split()) if element is not None else ""


def load_clinical_annotations(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for _, status in ET.iterparse(path, events=("end",)):
        if _tag(status) != "HPODisorderSetStatus":
            continue
        disorder = _child(status, "Disorder")
        disease_id = canonical_disease_id(_text(disorder, "OrphaCode"))
        if disease_id is None:
            raise ValueError("Orphanet clinical record has invalid OrphaCode")
        for index, association in enumerate(
            _children(
                _child(disorder, "HPODisorderAssociationList"), "HPODisorderAssociation"
            )
        ):
            frequency = _text(association, "HPOFrequency", "Name")
            rows.append(
                {
                    "disease_id": disease_id,
                    "disease_name": _text(disorder, "Name"),
                    "disease_id_original": _text(disorder, "OrphaCode"),
                    "hpo_id_original": _text(association, "HPO", "HPOId"),
                    "hpo_name_original": _text(association, "HPO", "HPOTerm"),
                    "frequency": frequency or None,
                    "frequency_weight": frequency_weight(frequency),
                    "is_negative": is_negative(None, frequency),
                    "qualifier": "",
                    "diagnostic_criterion": _text(
                        association, "DiagnosticCriteria", "Name"
                    ),
                    "source_validation": _text(status, "ValidationStatus"),
                    "validation_date": _text(status, "ValidationDate"),
                    "source": _text(status, "Source"),
                    "source_database": "Orphanet",
                    "source_file": path.name,
                    "source_record_id": f"{status.get('id', disease_id)}:{index}",
                }
            )
        status.clear()
    return rows


def load_gene_associations(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for _, disorder in ET.iterparse(path, events=("end",)):
        if _tag(disorder) != "Disorder":
            continue
        disease_id = canonical_disease_id(_text(disorder, "OrphaCode"))
        if disease_id is None:
            raise ValueError("Orphanet gene record has invalid OrphaCode")
        for index, association in enumerate(
            _children(
                _child(disorder, "DisorderGeneAssociationList"),
                "DisorderGeneAssociation",
            )
        ):
            gene = _child(association, "Gene")
            if gene is None:
                raise ValueError("Orphanet association is missing Gene")
            key = gene.get("id", "").strip()
            refs: dict[str, list[str]] = {}
            for ref in _children(
                _child(gene, "ExternalReferenceList"), "ExternalReference"
            ):
                namespace, value = _text(ref, "Source"), _text(ref, "Reference")
                if namespace and value:
                    refs.setdefault(namespace, []).append(value)
            refs = {k: sorted(set(v)) for k, v in sorted(refs.items())}
            hgnc_values = refs.get("HGNC", [])
            reason = (
                "missing_hgnc"
                if not hgnc_values
                else "ambiguous_hgnc"
                if len(hgnc_values) > 1
                else "source_hgnc"
            )
            try:
                identity = canonical_gene(
                    hgnc_values[0] if len(hgnc_values) == 1 else None, key
                )
            except ValueError:
                identity = canonical_gene(None, key)
                reason = "invalid_hgnc"
            rows.append(
                {
                    "disease_id": disease_id,
                    "disease_id_original": _text(disorder, "OrphaCode"),
                    "disease_name": _text(disorder, "Name"),
                    **identity,
                    "source_gene_key": key,
                    "gene_symbol": _text(gene, "Symbol"),
                    "gene_name": _text(gene, "Name"),
                    "gene_type": _text(gene, "GeneType", "Name"),
                    "gene_xrefs": refs,
                    "gene_resolution_reason": reason,
                    "association_type": _text(
                        association, "DisorderGeneAssociationType", "Name"
                    ),
                    "association_status": _text(
                        association, "DisorderGeneAssociationStatus", "Name"
                    ),
                    "source_validation": _text(association, "SourceOfValidation"),
                    "source_database": "Orphanet",
                    "source_file": path.name,
                    "source_record_id": f"{disorder.get('id', disease_id)}:{index}",
                }
            )
        disorder.clear()
    return rows
