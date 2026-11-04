import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.etl.hpo_loader import load_knowledge_base
from app.etl.processed_store import save_knowledge_base


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build processed RARE_DX_AI knowledge files from raw data."
    )
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument(
        "--canonical-graph",
        action="store_true",
        help="Opt-in ORPHA/HPO/HGNC graph export; preserves legacy JSON files",
    )
    parser.add_argument("--manifest", type=Path, default=Path("manifests/sources.json"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    args = parser.parse_args()
    if args.canonical_graph:
        from app.etl.source_manifest import build_source_manifest
        from app.graph.dataset import build_canonical_graph, write_graph

        sources = build_source_manifest(
            args.raw_dir, repository_root=Path(__file__).resolve().parents[1]
        )
        hpo_dir, orphanet_dir = args.raw_dir / "hpo", args.raw_dir / "orphanet"
        gene_phenotypes = hpo_dir / "genes_to_phenotype.txt"
        graph = build_canonical_graph(
            hpo_dir / "hp.obo",
            hpo_dir / "phenotype.hpoa",
            orphanet_dir / "Clinical_signs_and_symptoms_in_rare_disease.xml",
            orphanet_dir / "genes_associated_with_rare_disease.xml",
            gene_phenotypes if gene_phenotypes.is_file() else None,
            seed=args.seed,
            val_fraction=args.validation_fraction,
            test_fraction=args.test_fraction,
            sources=sources,
        )
        write_graph(graph, args.output_dir / "graph")
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(
            json.dumps(
                sources, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
            )
            + "\n",
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "nodes": graph["graph_metadata"]["node_counts"],
                    "edges": graph["graph_metadata"]["relation_counts"],
                    "qc_passed": graph["qc_report"]["passed"],
                },
                sort_keys=True,
            )
        )
        if not graph["qc_report"]["passed"]:
            raise SystemExit(
                "Canonical graph QC failed; inspect qc_report.json before using outputs"
            )
        return
    kb = load_knowledge_base(
        hpo_obo_path=args.raw_dir / "hp.obo",
        phenotype_hpoa_path=args.raw_dir / "phenotype.hpoa",
        genes_to_phenotype_path=args.raw_dir / "genes_to_phenotype.txt",
    )
    save_knowledge_base(kb, args.output_dir)
    print(
        f"Processed knowledge base written: {len(kb.phenotypes)} phenotypes, {len(kb.disease_phenotypes)} disease-phenotype annotations, {len(kb.gene_phenotypes)} gene-phenotype annotations"
    )


if __name__ == "__main__":
    main()
