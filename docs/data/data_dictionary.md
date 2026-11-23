# Canonical graph data dictionary

Layer order: **raw data → cleaned table → mapping table → graph/model input**.

CSV NULL is an empty cell; booleans are `true`/`false`; array/dict cells are JSON, not ambiguous delimiter joins. Numeric values are finite. IDs remain strings. Empty tables retain contract headers, including provenance and identity/weight fields. Relation-specific files are the source of truth; unified views are derived exports.

| File / layer | Key fields and semantics |
|---|---|
| raw HPO/OBO/XML/TSV | immutable source IDs, assertions, dates and evidence |
| cleaned_disease_hpo.csv | raw fields, source_file/source_record_id, original IDs, qualifier, is_negative, raw_annotation, frequency_weight; includes external-namespace source rows |
| cleaned_disease_gene.csv | one source association per row; gene_xrefs, source_gene_key, identity status, exact type/status weights, evidence_score, weight_rule_version |
| cleaned_gene_phenotype.csv | source NCBI Gene-HPO rows with full raw_annotation, frequency/qualifier, is_negative and ingestion status; separate from canonical graph |
| mapped_disease_hpo.csv | ORPHA annotations plus original/canonical HPO and resolution method; includes negatives and unresolved for review |
| hpo_nodes.csv | node_id, node_type=Phenotype, name, definition, synonyms; active ontology universe |
| disease_nodes.csv | node_id, node_type=Disease, source_names, xrefs; union of phenotype/gene disease sets |
| gene_nodes.csv | node_id, node_type=Gene, source gene symbols/names/xrefs, canonical_status, rankable, resolution_basis |
| hpo_hpo_edges.csv | source_id child → target_id parent, relation=IS_A, source/version |
| disease_hpo_edges.csv | one positive canonical pair; frequency_weight nullable, frequency_conflict, source_database, evidence_count/evidence_provenance |
| disease_gene_edges.csv | one canonical pair; independent association_type/status arrays, evidence_score, gene_df, idf_raw/norm/penalty, evidence_score_idf, evidence_count, source_validation/database, rule versions and idf_scope=full_reference |
| gene_hpo_edges.csv | verified NCBI→HGNC mapped source pairs only; ASSOCIATED_PHENOTYPE; reference-only training eligibility |
| graph_nodes.csv / graph_edges.csv | union interoperability views; never input mock scaffold |
| negative_phenotype_annotations.csv | NOT/Excluded/zero-frequency assertions, including original IDs/source data; absent from positive graph |
| negative_gene_phenotype_annotations.csv | NOT/zero-frequency Gene-HPO assertions; preserved even when NCBI identity is unresolved; absent from positive graph |
| unresolved_hpo.csv | original ID, method/reason, source context; consider_only is not auto-remapped |
| unresolved_genes.csv | unresolved Orphanet locus nodes, rankable=false |
| gene_phenotype_unresolved.csv | unmapped/ambiguous NCBI IDs or invalid HPO assertions; no symbol-based merging |
| non_orpha_annotations.csv | external disease annotations without verified ORPHA mapping; retained, not canonical disease nodes |
| rejected_source_annotations.csv | malformed/rejected raw annotation fields and ingestion status; not silently discarded |
| graph_metadata.json / qc_report.json | graph/source/rule/split contract, counts, errors, audit distributions |
| splits/*.csv / idf_fit.json | train-fitted statistics and edge partitions; see split_policy.md |
| artifact_hashes.json | SHA-256 per emitted file; excludes itself |

## Duplicate projection

DH groups positive evidence by canonical disease/HPO pair, keeps every provenance record, and uses the maximum known frequency; all-missing stays NULL. `frequency_conflict=true` flags disagreement, not clinical resolution. Sources often share upstream annotations, so evidence_count is a record count, not independent evidence strength.

DG groups canonical disease/gene pairs before split, retains every source association, and takes maximum evidence_score. Association/status/source arrays are sorted distinct values; full original association records remain in evidence_provenance. IDF counts unique pairs, never evidence rows. Minimum penalty is 0.5; constant IDF ranges use normalized value 1.0. Unknown type/status fails rather than inventing a weight; missing status uses the inherited 0.5 policy. Heuristic weights are not clinical probabilities.
