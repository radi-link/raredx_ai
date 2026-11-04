import hashlib
from pathlib import Path

from app.etl.source_manifest import build_source_manifest


def test_manifest_hash_and_unknown_dates_are_honest(tmp_path: Path) -> None:
    raw = tmp_path / "data" / "raw"
    (raw / "hpo").mkdir(parents=True)
    content = "format-version: 1.2\ndata-version: hp/releases/synthetic\n"
    (raw / "hpo" / "hp.obo").write_text(content)
    result = build_source_manifest(raw, repository_root=tmp_path)
    rows = {row["source_name"]: row for row in result["sources"]}
    assert (
        rows["HPO_ontology"]["sha256"] == hashlib.sha256(content.encode()).hexdigest()
    )
    assert rows["HPO_ontology"]["release_version"] == "hp/releases/synthetic"
    assert rows["HPO_ontology"]["acquisition_date"] is None
    assert rows["HPO_ontology"]["local_relative_path"] == "data/raw/hpo/hp.obo"
    assert rows["Orphanet_alignments"]["availability_status"] == "NOT_AVAILABLE"
    assert (
        rows["HGNC_NCBI_mapping"]["reproducibility_status"]
        == "REQUIRED_FOR_FUTURE_CANONICALIZATION"
    )
    assert rows["OMIM"]["availability_status"] == "NOT_AVAILABLE"
    assert rows["OMIM"]["source_role"] == "NOT_USED_IN_V1"
