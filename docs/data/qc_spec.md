# Graph QC specification

Flow: **raw data → cleaned table → mapping table → graph/model input**. Inspect both source audits and final graph integrity; passing final-edge checks alone must not hide rejected source ontology links.

Errors make qc_report.passed false and canonical CLI exit nonzero:
- duplicate/null node ID or invalid namespace;
- missing edge endpoint, invalid relation/type direction, self-loop;
- exact duplicate or canonical relation/pair duplicate;
- nonnumeric/nonfinite/negative weight or bounded weight outside [0,1]; IDF raw may exceed 1;
- negative assertion inside positive HAS_PHENOTYPE/ASSOCIATED_PHENOTYPE;
- nonactive HPO graph endpoint, invalid source IS_A reference, ontology cycle;
- missing DG IDF join;
- cross-split pair/reverse leakage, duplicate task edges, known-positive negative collision, wrong train-only fit metadata.

Audits report HPO alt/replacement/unresolved reasons, rejected raw rows, unresolved gene status, frequency distributions/conflicts, association type/status distributions, score ranges, duplicate counts before/after pair projection, negative annotations and source coverage. Unresolved consider-only records remain audited and cannot enter positive edges. Unknown frequency is NULL, not a QC error or imputed default.

## Checks

```bash
.venv/bin/python -m pytest -q
.venv/bin/python scripts/build_processed.py --canonical-graph
```

Small synthetic fixture tests cover malformed ID/NOT spacing, alt/replaced_by/consider/cycles, source provenance, invalid ontology modifiers/references, union disease nodes, duplicate scoring, train-only stats, reverse tying, full-positive negative exclusion, deterministic export and legacy JSON preservation. Real graph QC and repeated file hashes provide dataset-build evidence, not model performance or clinical validity.
