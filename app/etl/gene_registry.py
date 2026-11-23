"""Strict, offline HGNC/NCBI identity mapping; no graph mutation or symbol inference."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import re
from collections import Counter, defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import Any

POLICY_VERSION = "hgnc_ncbi_reciprocal_v1"
SOURCE_FILES = (
    "hgnc_complete_set.txt",
    "withdrawn.txt",
    "Homo_sapiens.gene_info.gz",
    "gene_history.gz",
)
XREF_FIELDS = [
    "hgnc_id",
    "ncbi_gene_id",
    "status",
    "hgnc_symbol",
    "ncbi_symbol",
    "source",
    "source_version",
]
AUDIT_FIELDS = [
    "source",
    "source_record_id",
    "hgnc_id",
    "ncbi_gene_id",
    "symbol",
    "status",
    "reason",
    "replacement_ids",
    "ncbi_hgnc_xrefs",
    "raw_record",
]
COVERAGE_FIELDS = [
    "source",
    "identifier",
    "mapped_hgnc_id",
    "mapped_ncbi_gene_id",
    "mapping_status",
    "reason",
    "annotation_rows",
]


def _rows(
    path: Path,
    required: set[str],
    *,
    delimiter: str = "\t",
    optional_trailing: frozenset[str] = frozenset(),
    tax_id: str | None = None,
) -> Iterator[dict[str, str]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        reader.fieldnames = [
            name.lstrip("#").strip().split(" (", 1)[0].lower().replace(" ", "_")
            for name in (reader.fieldnames or [])
        ]
        reader.fieldnames = [
            "symbol" if name == "withdrawn_symbol" else name
            for name in reader.fieldnames
        ]
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(f"{path.name}: missing required columns {sorted(missing)}")
        if len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise ValueError(f"{path.name}: duplicate column names")
        for row in reader:
            if tax_id is not None and (row.get("tax_id") or "").strip() != tax_id:
                continue
            if None in row or any(
                value is None and key not in optional_trailing
                for key, value in row.items()
            ):
                raise ValueError(
                    f"{path.name}: malformed TSV row at line {reader.line_num}"
                )
            yield {key: (value or "").strip() for key, value in row.items()}


def _hgnc(value: str) -> str | None:
    match = re.fullmatch(r"(?:HGNC:)?([0-9]+)", value)
    return f"HGNC:{int(match[1])}" if match and int(match[1]) > 0 else None


def _ncbi(value: str) -> str | None:
    return (
        str(int(value)) if re.fullmatch(r"[0-9]+", value) and int(value) > 0 else None
    )


def _json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        allow_nan=False,
        separators=(",", ":"),
    )


def _digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def build_gene_mapping(
    hgnc_path: Path,
    gene_info_path: Path,
    gene_history_path: Path,
    withdrawn_path: Path,
    *,
    gene_nodes_path: Path | None = None,
    gene_annotations_path: Path | None = None,
) -> dict[str, Any]:
    """Accept unique reciprocal current-human pairs; retain every rejected source row."""
    current: dict[str, dict[str, Any]] = {}
    reverse: dict[str, set[str]] = defaultdict(set)
    for row in _rows(
        gene_info_path, {"tax_id", "geneid", "symbol", "dbxrefs"}, tax_id="9606"
    ):
        if row["tax_id"] != "9606":
            continue
        ncbi_id = _ncbi(row["geneid"])
        if ncbi_id is None:
            raise ValueError("Invalid human NCBI GeneID in gene_info")
        if ncbi_id in current:
            raise ValueError(f"duplicate human GeneID in gene_info: {ncbi_id}")
        refs: set[str] = set()
        malformed_refs: list[str] = []
        for raw_ref in row["dbxrefs"].split("|"):
            ref = raw_ref.strip()
            if ref.startswith("HGNC:"):
                hgnc_id = _hgnc(ref.removeprefix("HGNC:"))
                if hgnc_id:
                    refs.add(hgnc_id)
                    reverse[hgnc_id].add(ncbi_id)
                else:
                    malformed_refs.append(ref)
        current[ncbi_id] = {
            "symbol": row["symbol"],
            "hgnc_ids": refs,
            "malformed_hgnc_xrefs": malformed_refs,
        }
    if not current:
        raise ValueError("gene_info contains no current human GeneID records")

    history: dict[str, list[dict[str, str]]] = defaultdict(list)
    history_rows = 0
    for row in _rows(
        gene_history_path,
        {"tax_id", "geneid", "discontinued_geneid", "discontinue_date"},
        tax_id="9606",
    ):
        if row["tax_id"] != "9606":
            continue
        identifier = _ncbi(row["discontinued_geneid"])
        if identifier is None:
            raise ValueError("Invalid human discontinued GeneID in gene_history")
        history[identifier].append(row)
        history_rows += 1

    hgnc_rows = list(_rows(hgnc_path, {"hgnc_id", "symbol", "status", "entrez_id"}))
    if not hgnc_rows:
        raise ValueError("HGNC registry contains no records")
    withdrawn = list(
        _rows(
            withdrawn_path,
            {"hgnc_id", "symbol", "status", "merged_into_report(s)"},
            optional_trailing=frozenset({"merged_into_report(s)"}),
        )
    )
    withdrawn_ids = {_hgnc(row["hgnc_id"]) for row in withdrawn}
    by_hgnc: dict[str, list[dict[str, str]]] = defaultdict(list)
    by_ncbi: dict[str, set[str]] = defaultdict(set)
    for row in hgnc_rows:
        hid, nid = _hgnc(row["hgnc_id"]), _ncbi(row["entrez_id"])
        if hid:
            by_hgnc[hid].append(row)
        if hid and nid and row["status"] == "Approved":
            by_ncbi[nid].add(hid)

    mapping: list[dict[str, str]] = []
    audit: list[dict[str, str]] = []
    rejection_by_hgnc: dict[str, str] = {}
    rejection_by_ncbi: dict[str, set[str]] = defaultdict(set)
    for index, row in enumerate(hgnc_rows, start=2):
        hid, nid = _hgnc(row["hgnc_id"]), _ncbi(row["entrez_id"])
        info = current.get(nid or "")
        refs = info["hgnc_ids"] if info else set()
        reason = (
            "invalid_hgnc_id"
            if hid is None
            else "conflicting_hgnc_records"
            if len(by_hgnc[hid]) != 1
            else "hgnc_not_approved"
            if row["status"] != "Approved"
            else "hgnc_withdrawn_registry"
            if hid in withdrawn_ids
            else "missing_ncbi_gene_id"
            if row["entrez_id"] in {"", "-"}
            else "invalid_ncbi_gene_id"
            if nid is None
            else "ambiguous_hgnc_ncbi"
            if len(by_ncbi[nid]) != 1
            else "retired_ncbi_gene_id"
            if nid in history
            else "ncbi_gene_not_current_human"
            if info is None
            else "malformed_ncbi_hgnc_xref"
            if info["malformed_hgnc_xrefs"]
            else "ncbi_hgnc_xref_missing"
            if not refs
            else "ambiguous_ncbi_hgnc"
            if len(refs) != 1
            else "reciprocal_mismatch"
            if refs != {hid}
            else "ambiguous_reverse_ncbi"
            if reverse[hid] != {nid}
            else ""
        )
        if reason:
            replacements = sorted({r["geneid"] for r in history.get(nid or "", [])})
            audit.append(
                {
                    "source": "HGNC_complete",
                    "source_record_id": str(index),
                    "hgnc_id": hid or row["hgnc_id"],
                    "ncbi_gene_id": nid or row["entrez_id"],
                    "symbol": row["symbol"],
                    "status": row["status"],
                    "reason": reason,
                    "replacement_ids": "|".join(replacements),
                    "ncbi_hgnc_xrefs": "|".join(sorted(refs)),
                    "raw_record": _json(row),
                }
            )
            if hid:
                rejection_by_hgnc[hid] = reason
            if nid:
                rejection_by_ncbi[nid].add(reason)
            continue
        assert hid is not None and nid is not None and info is not None
        mapping.append(
            {
                "hgnc_id": hid,
                "ncbi_gene_id": nid,
                "status": "validated_reciprocal",
                "hgnc_symbol": row["symbol"],
                "ncbi_symbol": info["symbol"],
                "source": "HGNC_complete;NCBI_gene_info;NCBI_gene_history",
                "source_version": POLICY_VERSION,
            }
        )
    for index, row in enumerate(withdrawn, start=2):
        hid = _hgnc(row["hgnc_id"])
        audit.append(
            {
                "source": "HGNC_withdrawn",
                "source_record_id": str(index),
                "hgnc_id": hid or row["hgnc_id"],
                "ncbi_gene_id": "",
                "symbol": row["symbol"],
                "status": row["status"],
                "reason": "hgnc_withdrawn_registry",
                "replacement_ids": row["merged_into_report(s)"],
                "ncbi_hgnc_xrefs": "",
                "raw_record": _json(row),
            }
        )
        if hid:
            rejection_by_hgnc.setdefault(hid, "hgnc_withdrawn_registry")
    mapping.sort(key=lambda row: (row["hgnc_id"], row["ncbi_gene_id"]))
    audit.sort(key=lambda row: (row["source"], int(row["source_record_id"])))
    hgnc_to_ncbi = {row["hgnc_id"]: row["ncbi_gene_id"] for row in mapping}
    ncbi_to_hgnc = {row["ncbi_gene_id"]: row["hgnc_id"] for row in mapping}
    coverage: list[dict[str, Any]] = []
    coverage_qc: dict[str, Any] = {}
    coverage_inputs: dict[str, Any] = {}
    if gene_nodes_path is not None:
        node_rows = list(_rows(gene_nodes_path, {"node_id"}, delimiter=","))
        seen: set[str] = set()
        for row in node_rows:
            identifier = row["node_id"]
            if identifier in seen:
                raise ValueError(f"Duplicate v1 gene node: {identifier}")
            seen.add(identifier)
            hid = _hgnc(identifier)
            target = hgnc_to_ncbi.get(hid or "", "")
            coverage.append(
                {
                    "source": "v1_gene_nodes",
                    "identifier": identifier,
                    "mapped_hgnc_id": hid if target else "",
                    "mapped_ncbi_gene_id": target,
                    "mapping_status": "mapped" if target else "unresolved",
                    "reason": ""
                    if target
                    else rejection_by_hgnc.get(
                        hid or "",
                        "hgnc_not_in_registry" if hid else "noncanonical_gene_node",
                    ),
                    "annotation_rows": 1,
                }
            )
        mapped = sum(row["mapping_status"] == "mapped" for row in coverage)
        rankable = sum(row.get("rankable") == "true" for row in node_rows)
        coverage_qc["v1_gene_nodes"] = {
            "total": len(node_rows),
            "mapped": mapped,
            "unresolved": len(node_rows) - mapped,
            "rankable_nodes": rankable,
            "mapped_fraction": mapped / len(node_rows) if node_rows else None,
        }
        coverage_inputs["v1_gene_nodes"] = {
            "file_name": gene_nodes_path.name,
            "sha256": _digest(gene_nodes_path),
        }
    if gene_annotations_path is not None:
        counts: Counter[str] = Counter()
        for row in _rows(gene_annotations_path, {"ncbi_gene_id"}):
            counts[row["ncbi_gene_id"]] += 1
        for identifier, count in sorted(counts.items()):
            nid = _ncbi(identifier)
            target = ncbi_to_hgnc.get(nid or "", "")
            reasons = rejection_by_ncbi.get(nid or "", set())
            reason = (
                ""
                if target
                else "invalid_ncbi_gene_id"
                if nid is None
                else "retired_ncbi_gene_id"
                if nid in history
                else "ncbi_gene_not_current_human"
                if nid not in current
                else "|".join(sorted(reasons))
                if reasons
                else "no_approved_hgnc_forward_mapping"
            )
            coverage.append(
                {
                    "source": "HPO_genes_to_phenotype",
                    "identifier": identifier,
                    "mapped_hgnc_id": target,
                    "mapped_ncbi_gene_id": nid if target else "",
                    "mapping_status": "mapped" if target else "unresolved",
                    "reason": reason,
                    "annotation_rows": count,
                }
            )
        mapped_rows = sum(
            count
            for identifier, count in counts.items()
            if _ncbi(identifier) in ncbi_to_hgnc
        )
        mapped_ids = sum(_ncbi(identifier) in ncbi_to_hgnc for identifier in counts)
        total_rows = sum(counts.values())
        coverage_qc["HPO_genes_to_phenotype"] = {
            "total_rows": total_rows,
            "mapped_rows": mapped_rows,
            "unresolved_rows": total_rows - mapped_rows,
            "total_unique_ids": len(counts),
            "mapped_unique_ids": mapped_ids,
            "mapped_row_fraction": mapped_rows / total_rows if total_rows else None,
            "note": "Identity coverage only; phenotype validity, negative qualifiers and disease provenance are not evaluated. Gene-HPO remains excluded from the training graph.",
        }
        coverage_inputs["HPO_genes_to_phenotype"] = {
            "file_name": gene_annotations_path.name,
            "sha256": _digest(gene_annotations_path),
        }
    qc = {
        "policy_version": POLICY_VERSION,
        "passed": bool(mapping)
        and len(mapping) == len(hgnc_to_ncbi) == len(ncbi_to_hgnc),
        "tax_id": "9606",
        "hgnc_records": len(hgnc_rows),
        "approved_hgnc_records": sum(row["status"] == "Approved" for row in hgnc_rows),
        "current_human_ncbi_records": len(current),
        "human_history_records": history_rows,
        "withdrawn_registry_records": len(withdrawn),
        "accepted_pairs": len(mapping),
        "hgnc_complete_rejected_records": sum(
            row["source"] == "HGNC_complete" for row in audit
        ),
        "audit_reason_counts": dict(
            sorted(Counter(row["reason"] for row in audit).items())
        ),
        "coverage": coverage_qc,
        "coverage_inputs": coverage_inputs,
        "graph_modified": False,
        "gene_hpo_training_context": "quarantined_all",
    }
    return {
        "gene_id_xref": mapping,
        "gene_mapping_audit": audit,
        "gene_mapping_coverage": coverage,
        "gene_mapping_qc": qc,
    }


def build_snapshot(
    snapshot_dir: Path,
    output_dir: Path,
    *,
    gene_nodes_path: Path | None = None,
    gene_annotations_path: Path | None = None,
) -> dict[str, Any]:
    """Verify immutable inputs, then write a new output directory. Never overwrite."""
    if output_dir.exists():
        raise FileExistsError(f"Output already exists: {output_dir}")
    manifest_path = snapshot_dir / "source_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("manifest_version") != "gene_registry_sources_v1":
        raise ValueError("Unsupported registry source manifest version")
    for name in SOURCE_FILES:
        expected = manifest.get("sources", {}).get(name, {}).get("sha256")
        if not expected or _digest(snapshot_dir / name) != expected:
            raise ValueError(f"SHA-256 mismatch or missing hash: {name}")
    result = build_gene_mapping(
        snapshot_dir / "hgnc_complete_set.txt",
        snapshot_dir / "Homo_sapiens.gene_info.gz",
        snapshot_dir / "gene_history.gz",
        snapshot_dir / "withdrawn.txt",
        gene_nodes_path=gene_nodes_path,
        gene_annotations_path=gene_annotations_path,
    )
    if not result["gene_mapping_qc"]["passed"]:
        raise ValueError("Registry mapping QC failed")
    # The concrete source snapshot is distinct from the policy version.
    source_version = _digest(manifest_path)
    for row in result["gene_id_xref"]:
        row["source_version"] = source_version
    result["gene_mapping_qc"]["source_manifest_sha256"] = source_version
    output_dir.mkdir(parents=True, exist_ok=False)
    for name, fields in [
        ("gene_id_xref", XREF_FIELDS),
        ("gene_mapping_audit", AUDIT_FIELDS),
        ("gene_mapping_coverage", COVERAGE_FIELDS),
    ]:
        with (output_dir / f"{name}.csv").open(
            "x", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(result[name])
    for name, data in [
        ("gene_mapping_qc", result["gene_mapping_qc"]),
    ]:
        (output_dir / f"{name}.json").write_text(
            json.dumps(
                data, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
            )
            + "\n",
            encoding="utf-8",
        )
    (output_dir / "source_manifest.json").write_bytes(manifest_path.read_bytes())
    hashes = {p.name: _digest(p) for p in sorted(output_dir.iterdir()) if p.is_file()}
    (output_dir / "artifact_hashes.json").write_text(
        json.dumps(hashes, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result["gene_mapping_qc"]
