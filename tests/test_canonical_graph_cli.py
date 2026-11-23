import json
import shutil
import subprocess
import sys
from pathlib import Path

FIXTURES = Path(__file__).parent / "fixtures" / "graph_phase2"
ROOT = Path(__file__).resolve().parents[1]


def test_opt_in_canonical_cli_does_not_overwrite_legacy_json(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    (raw / "hpo").mkdir(parents=True)
    (raw / "orphanet").mkdir()
    for name in ["hp.obo", "phenotype.hpoa"]:
        shutil.copyfile(FIXTURES / name, raw / "hpo" / name)
    for src, target in [
        ("clinical.xml", "Clinical_signs_and_symptoms_in_rare_disease.xml"),
        ("gene.xml", "genes_associated_with_rare_disease.xml"),
    ]:
        shutil.copyfile(FIXTURES / src, raw / "orphanet" / target)
    output = tmp_path / "processed"
    output.mkdir()
    legacy = output / "phenotypes.json"
    legacy.write_text("legacy sentinel")
    manifest = tmp_path / "sources.json"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/build_processed.py"),
            "--canonical-graph",
            "--raw-dir",
            str(raw),
            "--output-dir",
            str(output),
            "--manifest",
            str(manifest),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert legacy.read_text() == "legacy sentinel"
    assert (output / "graph" / "graph_nodes.csv").is_file()
    assert json.loads((output / "graph" / "qc_report.json").read_text())["passed"]
    assert json.loads(manifest.read_text())["sources"]
