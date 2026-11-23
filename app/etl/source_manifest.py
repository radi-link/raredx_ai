"""Local source metadata, hashes and explicit availability; no downloads or inferred dates."""

from __future__ import annotations

import hashlib
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

SOURCE_SPECS = [
    (
        "HPO_ontology",
        "hpo/hp.obo",
        "structural_prior",
        "HP",
        "http://purl.obolibrary.org/obo/hp.obo",
    ),
    (
        "HPO_annotations",
        "hpo/phenotype.hpoa",
        "disease_phenotype",
        "ORPHA;external_disease_namespaces",
        "https://purl.obolibrary.org/obo/hp/hpoa/phenotype.hpoa",
    ),
    (
        "HPO_gene_annotations",
        "hpo/genes_to_phenotype.txt",
        "gene_phenotype_reference",
        "NCBI_Gene",
        "https://purl.obolibrary.org/obo/hp/hpoa/genes_to_phenotype.txt",
    ),
    (
        "Orphanet_clinical",
        "orphanet/Clinical_signs_and_symptoms_in_rare_disease.xml",
        "disease_phenotype",
        "ORPHA;HP",
        None,
    ),
    (
        "Orphanet_genes",
        "orphanet/genes_associated_with_rare_disease.xml",
        "disease_gene",
        "ORPHA;HGNC;ORPHANET_LOCUS",
        None,
    ),
    (
        "Orphanet_alignments",
        "orphanet/rare_disease_alignments.xml",
        "disease_cross_reference_not_ingested_in_v1",
        "ORPHA;OMIM;MONDO;UMLS;ICD",
        None,
    ),
    (
        "HGNC_NCBI_mapping",
        "hgnc/hgnc_ncbi_mapping.tsv",
        "future_gene_canonicalization",
        "HGNC;NCBI_Gene",
        None,
    ),
    ("OMIM", "omim/omim_source.txt", "NOT_USED_IN_V1", "OMIM", None),
    (
        "DisGeNET",
        "disgenet/disease_gene_source.tsv",
        "NOT_USED_IN_V1",
        "external_disease;gene",
        None,
    ),
]


def _metadata(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "release_version": None,
        "source_date": None,
        "license": None,
    }
    if path.suffix == ".xml":
        for event, element in ET.iterparse(path, events=("start", "end")):
            tag = element.tag.rsplit("}", 1)[-1]
            if event == "start" and tag == "JDBOR":
                result.update(
                    release_version=element.get("version"),
                    source_date=element.get("date"),
                )
            if event == "end" and tag == "Licence":
                result["license"] = {
                    child.tag.rsplit("}", 1)[-1]: child.text for child in element
                }
                break
    else:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line.startswith("data-version: "):
                    result["release_version"] = line.removeprefix("data-version: ")
                elif line.startswith("#version:"):
                    result["release_version"] = line.split(":", 1)[1].strip()
                if "terms:license " in line:
                    result["license"] = {
                        "url": line.split("terms:license ", 1)[1].strip(),
                        "name": "HPO terms:license;see upstream terms",
                    }
                if line.startswith("[Term]") or (
                    path.suffix != ".obo" and not line.startswith("#")
                ):
                    break
    version_date = re.search(
        r"[0-9]{4}-[0-9]{2}-[0-9]{2}", result["release_version"] or ""
    )
    if result["source_date"] is None and version_date:
        result["source_date"] = version_date[0]
    return result


def build_source_manifest(raw_dir: Path, *, repository_root: Path) -> dict[str, Any]:
    sources = []
    for name, relative, role, namespace, url in SOURCE_SPECS:
        path = raw_dir / relative
        try:
            local_relative = str(path.resolve().relative_to(repository_root.resolve()))
            path_base = "repository"
        except ValueError:
            local_relative, path_base = relative, "raw_directory"
        available = path.is_file()
        metadata = (
            _metadata(path)
            if available
            else {"release_version": None, "source_date": None, "license": None}
        )
        notes = []
        if url:
            notes.append(
                "URL recorded from existing team download script; acquisition origin/date not recoverable from bytes."
            )
        if metadata["license"] is None:
            notes.append(
                "License not specified in local file metadata; verify source terms before redistribution."
            )
        if name == "HPO_gene_annotations":
            notes.append(
                "No release/date header in local TSV; do not infer a matching HPO release. NCBI IDs resolve only through direct Orphanet external references. Reference edges quarantined from DG training."
            )
        if name == "Orphanet_alignments":
            notes.append(
                "Legacy xref CSV is audit/reference only; absent raw XML is not fully reproducible. No xref-based disease identity merging in v1."
            )
        if name == "HGNC_NCBI_mapping":
            reproducibility = (
                "REQUIRED_FOR_FUTURE_CANONICALIZATION"
                if not available
                else "AVAILABLE_NOT_INGESTED_IN_V1"
            )
        elif role == "NOT_USED_IN_V1" or name == "Orphanet_alignments":
            reproducibility = (
                "NOT_USED_IN_V1" if available else "NOT_REPRODUCIBLE_RAW_MISSING"
            )
        else:
            reproducibility = (
                "REPRODUCIBLE_FROM_LOCAL_BYTES"
                if available
                else "NOT_REPRODUCIBLE_RAW_MISSING"
            )
        digest = None
        if available:
            with path.open("rb") as handle:
                digest = hashlib.file_digest(handle, "sha256").hexdigest()
        sources.append(
            {
                "source_name": name,
                "file_name": path.name,
                "source_role": role,
                **metadata,
                "acquisition_date": None,
                "canonical_id_namespace": namespace,
                "upstream_url": url,
                "local_relative_path": local_relative,
                "path_base": path_base,
                "sha256": digest,
                "availability_status": "AVAILABLE" if available else "NOT_AVAILABLE",
                "reproducibility_status": reproducibility,
                "notes": notes,
            }
        )
    return {"manifest_version": "sources_v1", "sources": sources}
