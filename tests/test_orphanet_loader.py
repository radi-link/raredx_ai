from pathlib import Path

from app.etl.orphanet_loader import load_clinical_annotations, load_gene_associations

FIXTURES = Path(__file__).parent / "fixtures" / "graph_phase2"


def test_clinical_retains_negative_and_source_evidence() -> None:
    rows = load_clinical_annotations(FIXTURES / "clinical.xml")
    assert len(rows) == 4
    assert rows[0]["disease_id"] == "ORPHA:1"
    assert rows[0]["hpo_id_original"] == "HP:0000022"
    assert rows[0]["source_validation"] == "Validated"
    assert rows[0]["source"] == "Synthetic publication"
    assert rows[1]["is_negative"] is True
    assert rows[3]["frequency_weight"] is None


def test_gene_retains_distinct_associations_and_unresolved_locus() -> None:
    rows = load_gene_associations(FIXTURES / "gene.xml")
    assert len(rows) == 3
    assert rows[0]["gene_id"] == "HGNC:10"
    assert rows[0]["gene_xrefs"]["Entrez"] == ["101"]
    assert rows[0]["source_validation"] == "PMID:synthetic1"
    assert rows[2]["gene_id"] == "ORPHANET_LOCUS:g2"
    assert rows[2]["rankable"] is False
    assert rows[2]["source_gene_key"] == "g2"


def test_missing_locus_key_is_not_guessed_from_symbol(tmp_path: Path) -> None:
    import pytest

    raw = (FIXTURES / "gene.xml").read_text().replace('Gene id="g2"', "Gene")
    path = tmp_path / "gene.xml"
    path.write_text(raw)
    with pytest.raises(ValueError, match="source locus key"):
        load_gene_associations(path)
