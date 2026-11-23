from pathlib import Path

import pytest

from app.etl.frequency import frequency_weight
from app.etl.id_mapping import canonical_disease_id, canonical_gene, load_hpo_catalog

FIXTURES = Path(__file__).parent / "fixtures" / "graph_phase2"


@pytest.mark.parametrize(
    ("raw", "canonical", "method"),
    [
        ("HP:0000002", "HP:0000002", "active"),
        ("HP:0000022", "HP:0000002", "alt_id"),
        ("HP:0000004", "HP:0000002", "replaced_by"),
    ],
)
def test_hpo_resolution(raw: str, canonical: str, method: str) -> None:
    result = load_hpo_catalog(FIXTURES / "hp.obo").resolve(raw)
    assert result.canonical_id == canonical
    assert result.original_id == raw
    assert result.method == method


@pytest.mark.parametrize(
    ("raw", "reason"),
    [
        ("HP:0000005", "consider_only"),
        ("HP:0000006", "replacement_cycle"),
        ("HP:9999999", "unknown_id"),
    ],
)
def test_unresolved_hpo_is_auditable(raw: str, reason: str) -> None:
    result = load_hpo_catalog(FIXTURES / "hp.obo").resolve(raw)
    assert result.canonical_id is None
    assert result.reason == reason


@pytest.mark.parametrize(
    ("raw", "want"),
    [
        ("58", "ORPHA:58"),
        ("ORPHA:058", "ORPHA:58"),
        ("OMIM:58", None),
        ("nan", None),
        ("", None),
    ],
)
def test_disease_namespace(raw: str, want: str | None) -> None:
    assert canonical_disease_id(raw) == want


def test_gene_resolution_does_not_infer_from_symbol() -> None:
    assert canonical_gene("HGNC:10", "g1")["gene_id"] == "HGNC:10"
    unresolved = canonical_gene(None, "g2")
    assert unresolved == {
        "gene_id": "ORPHANET_LOCUS:g2",
        "canonical_status": "unresolved",
        "rankable": False,
    }
    with pytest.raises(ValueError):
        canonical_gene("GENEA", "g1")


@pytest.mark.parametrize(
    ("raw", "want"),
    [
        ("Frequent (79-30%)", 0.545),
        ("HP:0040283", 0.17),
        ("Excluded (0%)", 0.0),
        ("", None),
        ("unknown", None),
        ("1/2", 0.5),
        ("25%", 0.25),
        ("1/0", None),
    ],
)
def test_frequency_does_not_impute_missing(raw: str, want: float | None) -> None:
    assert frequency_weight(raw) == want


@pytest.mark.parametrize("qualifier", [" NOT ", "NOT| OTHER", " OTHER | NOT "])
def test_negative_qualifier_whitespace_is_not_positive(qualifier: str) -> None:
    from app.etl.frequency import is_negative

    assert is_negative(qualifier, None)


def test_obo_is_a_modifier_is_not_part_of_parent_identifier(tmp_path: Path) -> None:
    source = tmp_path / "hp.obo"
    source.write_text(
        (FIXTURES / "hp.obo")
        .read_text()
        .replace(
            "is_a: HP:0000001 ! Synthetic root",
            'is_a: HP:0000001 {xref="PMID:synthetic"} ! Synthetic root',
        )
    )
    catalog = load_hpo_catalog(source)
    assert catalog.terms["HP:0000002"].parents == ("HP:0000001",)
