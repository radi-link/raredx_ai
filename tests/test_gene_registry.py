"""Registry contracts: identity must be explicit, current, human and reciprocal."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from app.etl.gene_registry import build_gene_mapping, build_snapshot

ROOT = Path(__file__).resolve().parents[1]


def tsv(path: Path, fields: list[str], rows: list[list[str]]) -> Path:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(fields)
        writer.writerows(rows)
    return path


def inputs(
    tmp_path: Path,
    *,
    hgnc: list[list[str]] | None = None,
    info: list[list[str]] | None = None,
    history: list[list[str]] | None = None,
    withdrawn: list[list[str]] | None = None,
) -> dict[str, Path]:
    return {
        "hgnc_path": tsv(
            tmp_path / "hgnc.txt",
            ["hgnc_id", "symbol", "status", "entrez_id"],
            hgnc if hgnc is not None else [["HGNC:1", "OLD_LABEL", "Approved", "10"]],
        ),
        "gene_info_path": tsv(
            tmp_path / "info.gz",
            ["#tax_id", "GeneID", "Symbol", "dbXrefs"],
            info
            if info is not None
            else [["9606", "10", "NEW_LABEL", "MIM:100|HGNC:HGNC:1"]],
        ),
        "gene_history_path": tsv(
            tmp_path / "history.gz",
            [
                "#tax_id",
                "GeneID",
                "Discontinued_GeneID",
                "Discontinued_Symbol",
                "Discontinue_Date",
            ],
            history or [],
        ),
        "withdrawn_path": tsv(
            tmp_path / "withdrawn.txt",
            ["HGNC_ID", "SYMBOL", "STATUS", "MERGED_INTO_REPORT(S)"],
            withdrawn or [],
        ),
    }


def test_explicit_reciprocal_ids_override_symbol_differences(tmp_path: Path) -> None:
    result = build_gene_mapping(**inputs(tmp_path))
    assert [(r["hgnc_id"], r["ncbi_gene_id"]) for r in result["gene_id_xref"]] == [
        ("HGNC:1", "10")
    ]
    assert result["gene_mapping_qc"]["passed"] is True
    assert result["gene_id_xref"][0]["status"] == "validated_reciprocal"


@pytest.mark.parametrize(
    ("patch", "reason"),
    [
        ({"hgnc": [["HGNC:1", "SAME", "Entry Withdrawn", "10"]]}, "hgnc_not_approved"),
        ({"hgnc": [["HGNC:1", "SAME", "Approved", ""]]}, "missing_ncbi_gene_id"),
        ({"hgnc": [["HGNC:1", "SAME", "Approved", "10.0"]]}, "invalid_ncbi_gene_id"),
        ({"hgnc": [["bad", "SAME", "Approved", "10"]]}, "invalid_hgnc_id"),
        (
            {
                "info": [
                    ["10090", "10", "SAME", "HGNC:HGNC:1"],
                    ["9606", "20", "OTHER", "HGNC:HGNC:2"],
                ]
            },
            "ncbi_gene_not_current_human",
        ),
        ({"info": [["9606", "10", "SAME", "MIM:123"]]}, "ncbi_hgnc_xref_missing"),
        ({"info": [["9606", "10", "SAME", "HGNC:HGNC:2"]]}, "reciprocal_mismatch"),
        (
            {"info": [["9606", "10", "SAME", "HGNC:HGNC:1|HGNC:HGNC:2"]]},
            "ambiguous_ncbi_hgnc",
        ),
        (
            {"history": [["9606", "20", "10", "SAME", "20260101"]]},
            "retired_ncbi_gene_id",
        ),
    ],
)
def test_unsafe_identity_is_audited_without_mapping(
    tmp_path: Path, patch: dict[str, list[list[str]]], reason: str
) -> None:
    result = build_gene_mapping(**inputs(tmp_path, **patch))
    assert result["gene_id_xref"] == []
    assert reason in {r["reason"] for r in result["gene_mapping_audit"]}


def test_many_hgnc_to_one_ncbi_rejects_all_candidates(tmp_path: Path) -> None:
    result = build_gene_mapping(
        **inputs(
            tmp_path,
            hgnc=[["HGNC:1", "A", "Approved", "10"], ["HGNC:2", "B", "Approved", "10"]],
        )
    )
    assert result["gene_id_xref"] == []
    assert {r["reason"] for r in result["gene_mapping_audit"]} == {
        "ambiguous_hgnc_ncbi"
    }


def test_conflicting_duplicate_hgnc_cannot_be_resolved_by_row_order(
    tmp_path: Path,
) -> None:
    result = build_gene_mapping(
        **inputs(
            tmp_path,
            hgnc=[["HGNC:1", "A", "Approved", "10"], ["HGNC:1", "A", "Approved", "20"]],
        )
    )
    assert result["gene_id_xref"] == []
    assert {r["reason"] for r in result["gene_mapping_audit"]} == {
        "conflicting_hgnc_records"
    }


def test_reverse_many_ncbi_to_one_hgnc_rejects_even_one_matching_forward(
    tmp_path: Path,
) -> None:
    result = build_gene_mapping(
        **inputs(
            tmp_path,
            info=[
                ["9606", "10", "A", "HGNC:HGNC:1"],
                ["9606", "20", "A", "HGNC:HGNC:1"],
            ],
        )
    )
    assert result["gene_id_xref"] == []
    assert result["gene_mapping_audit"][0]["reason"] == "ambiguous_reverse_ncbi"


def test_duplicate_ncbi_records_are_schema_failure(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="duplicate.*GeneID"):
        build_gene_mapping(
            **inputs(
                tmp_path,
                info=[
                    ["9606", "10", "A", "HGNC:HGNC:1"],
                    ["9606", "10", "B", "HGNC:HGNC:1"],
                ],
            )
        )


def test_withdrawn_merger_is_audit_only_and_history_taxa_are_scoped(
    tmp_path: Path,
) -> None:
    result = build_gene_mapping(
        **inputs(
            tmp_path,
            history=[["10090", "20", "10", "MOUSE", "20260101"]],
            withdrawn=[["HGNC:9", "OLD", "Merged/Split", "HGNC:1|HGNC:2"]],
        )
    )
    assert len(result["gene_id_xref"]) == 1
    audit = result["gene_mapping_audit"][0]
    assert audit["reason"] == "hgnc_withdrawn_registry"
    assert audit["replacement_ids"] == "HGNC:1|HGNC:2"


def test_missing_required_columns_fail_closed(tmp_path: Path) -> None:
    paths = inputs(tmp_path)
    tsv(paths["hgnc_path"], ["hgnc_id", "symbol", "entrez_id"], [["HGNC:1", "A", "10"]])
    with pytest.raises(ValueError, match="status"):
        build_gene_mapping(**paths)


def test_coverage_counts_rows_and_unique_ids_separately(tmp_path: Path) -> None:
    paths = inputs(tmp_path, history=[["9606", "10", "99", "A", "20260101"]])
    gene_nodes = tmp_path / "gene_nodes.csv"
    with gene_nodes.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["node_id", "rankable"])
        w.writerows(
            [["HGNC:1", "true"], ["HGNC:2", "true"], ["ORPHANET_LOCUS:x", "false"]]
        )
    annotations = tsv(
        tmp_path / "genes_to_phenotype.txt",
        ["ncbi_gene_id", "hpo_id", "disease_id"],
        [
            ["10", "HP:0000001", "OMIM:1"],
            ["10", "HP:0000002", "OMIM:2"],
            ["99", "HP:0000001", "OMIM:1"],
            ["999", "HP:0000001", "OMIM:1"],
        ],
    )
    result = build_gene_mapping(
        **paths, gene_nodes_path=gene_nodes, gene_annotations_path=annotations
    )
    qc = result["gene_mapping_qc"]["coverage"]
    assert qc["v1_gene_nodes"]["total"] == 3
    assert qc["v1_gene_nodes"]["mapped"] == 1
    assert qc["HPO_genes_to_phenotype"]["total_rows"] == 4
    assert qc["HPO_genes_to_phenotype"]["mapped_rows"] == 2
    assert qc["HPO_genes_to_phenotype"]["total_unique_ids"] == 3
    assert qc["HPO_genes_to_phenotype"]["mapped_unique_ids"] == 1
    rows = {
        r["identifier"]: r
        for r in result["gene_mapping_coverage"]
        if r["source"] == "HPO_genes_to_phenotype"
    }
    assert rows["99"]["reason"] == "retired_ncbi_gene_id"
    assert rows["99"]["mapped_hgnc_id"] == ""


def test_snapshot_cli_validates_hashes_and_refuses_existing_output(
    tmp_path: Path,
) -> None:

    paths = inputs(tmp_path)
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    names = {
        "hgnc_path": "hgnc_complete_set.txt",
        "gene_info_path": "Homo_sapiens.gene_info.gz",
        "gene_history_path": "gene_history.gz",
        "withdrawn_path": "withdrawn.txt",
    }
    sources = {}
    for key, name in names.items():
        payload = paths[key].read_bytes()
        (snapshot / name).write_bytes(payload)
        sources[name] = {
            "sha256": hashlib.sha256(payload).hexdigest(),
            "upstream_url": "synthetic:" + name,
            "release_version": None,
        }
    (snapshot / "source_manifest.json").write_text(
        json.dumps({"manifest_version": "gene_registry_sources_v1", "sources": sources})
    )
    out = tmp_path / "result"
    build_snapshot(snapshot, out)
    assert (out / "gene_id_xref.csv").exists()
    assert json.loads((out / "gene_mapping_qc.json").read_text())["passed"]
    assert (out / "artifact_hashes.json").exists()
    qc = json.loads((out / "gene_mapping_qc.json").read_text())
    assert (
        hashlib.sha256((out / "source_manifest.json").read_bytes()).hexdigest()
        == qc["source_manifest_sha256"]
    )
    with pytest.raises(FileExistsError):
        build_snapshot(snapshot, out)
    first = {p.name: p.read_bytes() for p in out.iterdir()}
    build_snapshot(snapshot, tmp_path / "second")
    assert first == {p.name: p.read_bytes() for p in (tmp_path / "second").iterdir()}
    with (snapshot / "hgnc_complete_set.txt").open("a") as f:
        f.write("tampered\n")
    with pytest.raises(ValueError, match="SHA-256"):
        build_snapshot(snapshot, tmp_path / "invalid")
    assert not (tmp_path / "invalid").exists()


def test_cli_builds_separate_outputs_without_touching_existing_v1(
    tmp_path: Path,
) -> None:

    paths = inputs(tmp_path)
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    sources = {}
    for key, name in {
        "hgnc_path": "hgnc_complete_set.txt",
        "gene_info_path": "Homo_sapiens.gene_info.gz",
        "gene_history_path": "gene_history.gz",
        "withdrawn_path": "withdrawn.txt",
    }.items():
        b = paths[key].read_bytes()
        (snapshot / name).write_bytes(b)
        sources[name] = {"sha256": hashlib.sha256(b).hexdigest()}
    (snapshot / "source_manifest.json").write_text(
        json.dumps({"manifest_version": "gene_registry_sources_v1", "sources": sources})
    )
    sentinel = tmp_path / "v1"
    sentinel.mkdir()
    (sentinel / "graph.csv").write_text("preserve")
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/build_gene_mapping.py"),
            "--snapshot-dir",
            str(snapshot),
            "--output-dir",
            str(tmp_path / "result"),
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert proc.returncode == 0, proc.stderr
    assert (sentinel / "graph.csv").read_text() == "preserve"
    assert json.loads(proc.stdout)["passed"] is True


def test_official_withdrawn_schema_allows_absent_optional_merger(
    tmp_path: Path,
) -> None:
    paths = inputs(tmp_path)
    paths["withdrawn_path"].write_text(
        "HGNC_ID\tSTATUS\tWITHDRAWN_SYMBOL\tMERGED_INTO_REPORT(S) (i.e HGNC_ID|SYMBOL|STATUS)\nHGNC:9\tEntry Withdrawn\tOLD\nHGNC:8\tMerged/Split\tMERGED\tHGNC:1|A|Approved\n"
    )
    result = build_gene_mapping(**paths)
    assert len(result["gene_id_xref"]) == 1
    rows = {r["hgnc_id"]: r for r in result["gene_mapping_audit"]}
    assert rows["HGNC:9"]["replacement_ids"] == ""
    assert rows["HGNC:8"]["replacement_ids"] == "HGNC:1|A|Approved"


def test_malformed_additional_ncbi_hgnc_xref_blocks_matching_pair(
    tmp_path: Path,
) -> None:
    result = build_gene_mapping(
        **inputs(tmp_path, info=[["9606", "10", "A", "HGNC:HGNC:1|HGNC:invalid"]])
    )
    assert result["gene_id_xref"] == []
    assert result["gene_mapping_audit"][0]["reason"] == "malformed_ncbi_hgnc_xref"


def test_zero_accepted_pairs_does_not_pass_snapshot_qc(tmp_path: Path) -> None:
    result = build_gene_mapping(**inputs(tmp_path, info=[["9606", "10", "A", "-"]]))
    assert result["gene_id_xref"] == []
    assert result["gene_mapping_qc"]["passed"] is False


def test_whitespace_does_not_hide_conflicting_reciprocal_reference(
    tmp_path: Path,
) -> None:
    result = build_gene_mapping(
        **inputs(tmp_path, info=[["9606", "10", "A", "HGNC:HGNC:1| HGNC:HGNC:2"]])
    )
    assert result["gene_id_xref"] == []
    assert result["gene_mapping_audit"][0]["reason"] == "ambiguous_ncbi_hgnc"
