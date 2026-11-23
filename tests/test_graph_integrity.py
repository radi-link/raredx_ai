import math
from pathlib import Path

import pytest

from app.graph.dataset import build_canonical_graph, write_graph
from app.graph.qc import validate_graph

FIXTURES = Path(__file__).parent / "fixtures" / "graph_phase2"


def build_fixture():
    return build_canonical_graph(
        FIXTURES / "hp.obo",
        FIXTURES / "phenotype.hpoa",
        FIXTURES / "clinical.xml",
        FIXTURES / "gene.xml",
    )


def test_graph_union_negatives_resolution_and_duplicate_provenance() -> None:
    result = build_fixture()
    assert {n["node_id"] for n in result["disease_nodes"]} == {
        "ORPHA:1",
        "ORPHA:8",
        "ORPHA:9",
    }
    assert {n["node_id"] for n in result["hpo_nodes"]} == {
        "HP:0000001",
        "HP:0000002",
        "HP:0000003",
    }
    dh = result["disease_hpo_edges"]
    assert {(r["source_id"], r["target_id"]) for r in dh} == {
        ("ORPHA:1", "HP:0000002"),
        ("ORPHA:9", "HP:0000003"),
    }
    combined = next(r for r in dh if r["source_id"] == "ORPHA:1")
    assert combined["frequency_weight"] == 0.545
    assert combined["evidence_count"] == 2
    assert len(combined["evidence_provenance"]) == 2
    assert len(result["negative_phenotype_annotations"]) == 2
    assert result["unresolved_hpo"][0]["reason"] == "consider_only"
    assert result["unresolved_genes"][0]["node_id"] == "ORPHANET_LOCUS:g2"
    dg = next(r for r in result["disease_gene_edges"] if r["source_id"] == "ORPHA:1")
    assert dg["evidence_score"] == 1.0
    assert dg["evidence_count"] == 2
    assert result["qc_report"]["passed"] is True
    assert result["split_qc"]["passed"] is True
    assert all(
        r["target_id"].startswith("HGNC:")
        for rows in result["splits"].values()
        for r in rows
    )
    assert all(
        r["relation"] != "ASSOCIATED_PHENOTYPE"
        for r in result["training_message_edges"]
    )


def test_byte_identical_exports_and_manifest_unchanged(tmp_path: Path) -> None:
    result = build_fixture()
    first, second = tmp_path / "first", tmp_path / "second"
    write_graph(result, first)
    write_graph(result, second)
    assert {
        p.relative_to(first): p.read_bytes() for p in first.rglob("*") if p.is_file()
    } == {
        p.relative_to(second): p.read_bytes() for p in second.rglob("*") if p.is_file()
    }
    assert (first / "hpo_nodes.csv").is_file()
    assert (first / "splits" / "train.csv").is_file()
    assert (first / "splits" / "idf_fit.json").is_file()


def test_qc_catches_dangling_invalid_negative_duplicate_and_nan() -> None:
    result = build_fixture()
    result["graph_nodes"].append(dict(result["graph_nodes"][0]))
    result["graph_edges"].append(
        {
            "source_id": "ORPHA:1",
            "target_id": "HP:9999999",
            "relation": "HAS_PHENOTYPE",
            "is_negative": True,
            "frequency_weight": math.nan,
        }
    )
    report = validate_graph(result)
    assert report["passed"] is False
    assert report["errors"]["duplicate_node_id"] == 1
    assert report["errors"]["target_node_missing"] == 1
    assert report["errors"]["negative_positive_edge"] == 1
    assert report["errors"]["invalid_weight"] == 1


def test_qc_catches_ontology_cycles_and_relation_type_mismatch() -> None:
    result = build_fixture()
    result["graph_edges"].append(
        {"source_id": "HP:0000001", "target_id": "HP:0000002", "relation": "IS_A"}
    )
    result["graph_edges"].append(
        {"source_id": "ORPHA:1", "target_id": "HGNC:10", "relation": "HAS_PHENOTYPE"}
    )
    report = validate_graph(result)
    assert report["errors"]["ontology_cycle"] > 0
    assert report["errors"]["invalid_relation_endpoints"] == 1


def test_invalid_source_is_a_reference_fails_qc(tmp_path: Path) -> None:
    obo = tmp_path / "hp.obo"
    obo.write_text(
        (FIXTURES / "hp.obo")
        .read_text()
        .replace("is_a: HP:0000001", "is_a: HP:9999999")
    )
    graph = build_canonical_graph(
        obo,
        FIXTURES / "phenotype.hpoa",
        FIXTURES / "clinical.xml",
        FIXTURES / "gene.xml",
    )
    assert graph["qc_report"]["passed"] is False
    assert graph["qc_report"]["errors"]["invalid_is_a_reference"] == 2


@pytest.mark.parametrize("missing_name", ["", "   "])
def test_rejected_hpoa_negative_raw_row_remains_in_audit(
    tmp_path: Path, missing_name: str
) -> None:
    hpoa = tmp_path / "phenotype.hpoa"
    hpoa.write_text(
        (FIXTURES / "phenotype.hpoa").read_text()
        + f"ORPHA:1\t{missing_name}\t NOT \tHP:0000002\t\tTAS\tPMID:missingname\n"
    )
    graph = build_canonical_graph(
        FIXTURES / "hp.obo", hpoa, FIXTURES / "clinical.xml", FIXTURES / "gene.xml"
    )
    assert len(graph["negative_phenotype_annotations"]) == 3
    assert len(graph["rejected_source_annotations"]) == 1
    assert (
        graph["rejected_source_annotations"][0]["raw_annotation"]["reference"]
        == "PMID:missingname"
    )


def test_gene_hpo_keeps_frequency_provenance_and_omits_negative(tmp_path: Path) -> None:
    genes = tmp_path / "genes_to_phenotype.txt"
    genes.write_text(
        "ncbi_gene_id\tgene_symbol\thpo_id\thpo_name\tfrequency\tdisease_id\n101\tGENEA\tHP:0000002\tSynthetic A\t1/2\tORPHA:1\n101\tGENEA\tHP:0000003\tSynthetic B\tHP:0040285\tORPHA:1\n999\tUNKNOWN\tHP:0000002\tSynthetic A\t-\tORPHA:1\n"
    )
    graph = build_canonical_graph(
        FIXTURES / "hp.obo",
        FIXTURES / "phenotype.hpoa",
        FIXTURES / "clinical.xml",
        FIXTURES / "gene.xml",
        genes,
    )
    assert len(graph["gene_hpo_edges"]) == 1
    row = graph["gene_hpo_edges"][0]
    assert row["source_id"] == "HGNC:10"
    assert row["target_id"] == "HP:0000002"
    assert row["evidence_provenance"][0]["raw_annotation"]["frequency"] == "1/2"
    assert len(graph["negative_gene_phenotype_annotations"]) == 1
    assert len(graph["cleaned_gene_phenotype"]) == 3
    assert graph["gene_phenotype_unresolved"][0]["gene_id"] == "999"
    assert all(
        row["relation"] != "ASSOCIATED_PHENOTYPE"
        for row in graph["training_message_edges"]
    )


def test_empty_exports_keep_contract_headers(tmp_path: Path) -> None:
    import csv

    result = build_fixture()
    write_graph(result, tmp_path)

    def fields(name):
        with (tmp_path / name).open() as handle:
            return next(csv.reader(handle))

    assert {"evidence_provenance", "training_context_eligible"} <= set(
        fields("gene_hpo_edges.csv")
    )
    assert {"raw_annotation", "reason", "gene_id"} <= set(
        fields("gene_phenotype_unresolved.csv")
    )
    assert {"raw_annotation", "is_negative"} <= set(
        fields("negative_gene_phenotype_annotations.csv")
    )
    assert {"evidence_score", "idf_scope", "split", "seed", "split_version"} <= set(
        fields("splits/test.csv")
    )


def test_empty_training_exports_keep_ablation_weight_contract(tmp_path: Path) -> None:
    import csv

    result = build_fixture()
    result["splits"]["train"] = []
    result["split_reverse"]["train"] = []
    write_graph(result, tmp_path)
    for mode in ("A", "B", "C"):
        with (tmp_path / "ablations" / f"{mode}_train.csv").open() as handle:
            fields = next(csv.reader(handle))
        assert "ablation" in fields
        assert ("edge_weight" in fields) is (mode != "C")
