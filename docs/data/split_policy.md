# Split and leakage policy

Flow: **raw data → cleaned table → mapping table → graph/model input**; task partitioning occurs after canonical-pair aggregation and before learned statistics.

Default task: transductive Disease-Gene link prediction, version dg_link_v1, seed 42, validation/test fractions 0.1/0.1. A local seeded RNG shuffles the sorted unique canonical ORPHA/HGNC pairs. Holdout sizes use floor(N*fraction); remaining pairs are train. Source evidence rows cannot cross partitions. Unresolved/nonrankable targets are retained in reference graph but excluded from task edges.

## Message passing

IS_A is structural prior and not randomly split. Disease-HPO is an input context, not a prediction target; it is not randomly split either. The utility rejects unsupported phenotype-target task contexts rather than claiming a safe random split. Default message graph = ontology + positive Disease-HPO + train DG. Validation/test DG positives are absent. Reverse DG rows inherit the same split/version/seed and identify their forward source via reverse_of.

All Gene-HPO edges are omitted from default DG message passing, even if mapped: original source disease associations may reveal heldout links. Reference exports preserve them with provenance. Target-derived IDF/provenance/source record fields are audit-only unless the experiment explicitly selects permitted train-fitted inputs. Disease/gene node universes are visible in transductive evaluation; this is not an inductive disease split.

## Statistics

Reference DG file uses idf_scope=full_reference. Split rows and B training ablation use idf_scope=train_only. Fit N, gene_df and normalization min/max on unique forward train pairs only; store the exact fit pairs and their hash in splits/idf_fit.json. Heldout transformation uses the frozen fit. Unseen genes have df=0 and their normalized value is clamped to [0,1]; constant/empty ranges use normalized 1.0. No validation/test statistics are fitted. The validator independently recomputes fit metadata and checks pairs, counts, bounds, hash and version.

## Negatives

Uniform sampling without replacement excludes the entire known positive set, including all splits. Only resolved HGNC genes are eligible targets. The utility checks capacity and never silently returns fewer requested samples. The builder records requested/generated/capacity and explicitly caps its example pool to available capacity. The negative pool represents unobserved pairs, not biologically verified negative labels; it is separate from negative phenotype annotations.

## Patient cases

Graph pair split and patient/case split are different experiments. There is no patient evaluation in this phase. Cases synthetically constructed from the same annotation graph used as model input can leak answers; define independent source/time/case partitions before claiming patient ranking quality. Any future target Disease-HPO experiment needs its own removal and source-overlap policy.
