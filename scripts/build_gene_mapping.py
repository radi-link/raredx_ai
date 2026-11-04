"""Build a separate registry mapping snapshot without invoking the graph builder."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.etl.gene_registry import build_snapshot


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Strict reciprocal HGNC–NCBI mapping; local snapshots only."
    )
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New directory; existing paths are rejected",
    )
    parser.add_argument(
        "--gene-nodes", type=Path, help="Read-only v1 gene_nodes.csv for coverage"
    )
    parser.add_argument(
        "--gene-annotations",
        type=Path,
        help="Read-only genes_to_phenotype.txt for identity coverage",
    )
    args = parser.parse_args()
    try:
        qc = build_snapshot(
            args.snapshot_dir,
            args.output_dir,
            gene_nodes_path=args.gene_nodes,
            gene_annotations_path=args.gene_annotations,
        )
    except (ValueError, OSError) as exc:
        parser.exit(1, f"Gene mapping failed: {exc}\n")
    print(json.dumps(qc, ensure_ascii=False, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
