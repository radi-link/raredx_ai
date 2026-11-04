# Data + Knowledge Graph Phase 2 implementation plan

Goal: build a deterministic, auditable ORPHA–HPO–HGNC graph and leakage-safe link experiment inputs without changing runtime ranking or training models.

Spec: the user's Phase 2 Data Contract supplied on 2026-10-07; this plan records implementation choices for that contract.

## Boundaries
- Existing app/etl HPO loader and legacy processed JSON flow remain available.
- Opt-in --canonical-graph on scripts/build_processed.py writes relation tables and audit/QC artifacts under ignored data/processed/graph.
- IDs: ORPHA:<integer>, active HP:<7 digits>, HGNC:<integer>; unresolved source loci are retained and not rankable.
- The ontology loader reuses the team's active-term parser and adds resolution metadata. Sources and full evidence rows stay auditable.
- Canonical DG pairs aggregate before split. Seed 42, validation/test fractions 0.1/0.1. Full-reference IDF is separate from train-only IDF.
- Default DG training context contains ontology and positive Disease-HPO only. Gene-HPO is retained in reference output but omitted from link-training message passing: upstream gene annotations may reveal heldout DG links.
- OMIM/MONDO/etc never become Phase 2 canonical nodes. Non-ORPHA annotations are retained in source/audit layers, not remapped without a verified cross-reference.
- No new project dependency, old source mutation, raw data commit or model training. Commit/push was deferred during implementation; team sharing was authorized on 2026-10-12.

## Tasks and ownership
- [x] ID resolution and frequency policy: app/etl/id_mapping.py, app/etl/frequency.py, tests/test_id_mapping.py; literal tests for alt/replacement/consider/cycles, malformed IDs, unknown frequency.
- [x] Orphanet ingestion: app/etl/orphanet_loader.py, tests/test_orphanet_loader.py; synthetic XML with scoped evidence, negative and fallback preservation.
- [x] Evidence / IDF: app/etl/evidence_scoring.py, tests/test_evidence_scoring.py; max aggregation, original evidence, exact formula, frozen train fit and unseen genes. Independent evidence lane.
- [x] Splits: app/graph/splits.py, tests/test_split_leakage.py; unique canonical pairs, reverse tying, full-positive negative exclusion, reject unsafe task context. Independent split lane.
- [x] Graph build/export/QC: app/graph/dataset.py, app/graph/qc.py, tests/test_graph_integrity.py; union nodes, positive filtering, dangling/cycle/finite/pair checks, deterministic real build.
- [x] Manifest and integration: app/etl/source_manifest.py, scripts/build_processed.py and offline frequency script; source hashes/version/license/unknowns, opt-in CLI, legacy compatibility.
- [x] Six data documents, source manifest, full-suite regression check and real-source reproducibility comparison.

## Verification
Write failing tests before production code for each lane, then targeted pytest. Run the whole repository suite on Python 3.12. Run changed-code lint and AST checks, git diff --check, QC and split validation. Build real graph twice to independent ignored/local locations and compare file hashes. Report environment limitations rather than inventing checks.

## Review focus
HPO consider-only and replacement cycles; conflicting source frequency and negative rows; malformed identifiers; empty/degenerate IDF; heldout DG recoverable through Gene-HPO provenance. Tests pin each behavior. Real datasets and synthetic patient labels are not performance evidence.


## Verified result — 2026-10-07

- Python 3.12 full suite: 154 passed; one external Starlette/httpx deprecation warning.
- Changed Python files: Ruff E4/E7/E9/F/I and AST parsing passed; git diff --check passed.
- Two independent real-source builds passed QC; all 36 output SHA-256 values were independently rehashed and matched. Source manifests were byte-identical.
- Nodes: HPO 19,810; Disease 5,996; Gene 4,552 (4,551 resolved, one nonrankable locus). Edges: IS_A 24,329; HAS_PHENOTYPE 115,098; ASSOCIATED_WITH 8,341; ASSOCIATED_PHENOTYPE zero.
- DG split: train 6,673 / validation 833 / test 833. Negative pool 8,339. Independent checks confirmed disjoint partitions, reverse tying, full-positive exclusion, exact train message pairs and train-fitted DF/IDF (N=3,488).
- HPO annotation resolution: 146 alt-ID conversions, zero replaced_by conversions in these real inputs, three consider-only records (HP:0025237). Negative Disease-HPO records 4,369; negative Gene-HPO records 2,961.
- Projection removed 115,198 duplicate DH evidence rows and 33 duplicate DG evidence rows while retaining provenance. Eighteen DH pairs retain frequency disagreement flags.
- Missing alignment XML and HGNC–NCBI mapping remain explicit. All 331,738 source Gene-HPO rows are retained in cleaned/unresolved audit tables; none can be safely joined using current source xrefs. Direct OMIM/DisGeNET raw inputs are absent. TSV release and unrecoverable acquisition/license metadata remain unknown.
- No model training, clinical-performance claim, raw dataset staging, commit or push. Legacy source directory remained read-only.

## Changed files

| Area | Files |
|---|---|
| New ETL | [frequency.py](../../app/etl/frequency.py), [id_mapping.py](../../app/etl/id_mapping.py), [orphanet_loader.py](../../app/etl/orphanet_loader.py), [evidence_scoring.py](../../app/etl/evidence_scoring.py), [source_manifest.py](../../app/etl/source_manifest.py) |
| New graph layer | [dataset.py](../../app/graph/dataset.py), [qc.py](../../app/graph/qc.py), [splits.py](../../app/graph/splits.py) |
| Existing scripts extended | [build_processed.py](../../scripts/build_processed.py), [add_frequency_weight .py](<../../scripts/HPO_Disease_Mapping/add_frequency_weight .py>) |
| Documentation | [data_source_manifest.md](data_source_manifest.md), [data_dictionary.md](data_dictionary.md), [id_mapping_policy.md](id_mapping_policy.md), [graph_schema.md](graph_schema.md), [split_policy.md](split_policy.md), [qc_spec.md](qc_spec.md), this plan |
| Manifest | [sources.json](../../manifests/sources.json) |
| Tests | [test_id_mapping.py](../../tests/test_id_mapping.py), [test_orphanet_loader.py](../../tests/test_orphanet_loader.py), [test_evidence_scoring.py](../../tests/test_evidence_scoring.py), [test_graph_integrity.py](../../tests/test_graph_integrity.py), [test_split_leakage.py](../../tests/test_split_leakage.py), [test_source_manifest.py](../../tests/test_source_manifest.py), [test_canonical_graph_cli.py](../../tests/test_canonical_graph_cli.py) |
| Synthetic fixtures | [hp.obo](../../tests/fixtures/graph_phase2/hp.obo), [phenotype.hpoa](../../tests/fixtures/graph_phase2/phenotype.hpoa), [clinical.xml](../../tests/fixtures/graph_phase2/clinical.xml), [gene.xml](../../tests/fixtures/graph_phase2/gene.xml) |


## Team handoff — 2026-10-12

User authorized committing and pushing the feature branch for the modeling team. Real datasets remain ignored and are prepared as a separate archive, not committed. The user designated the team Drive Processed_Data folder for upload; its new versioned subfolder is linked in the handoff document. [Modeling handoff](modeling_handoff.md) records required inputs, train/reference separation, CSV semantics, package verification and source regeneration.
