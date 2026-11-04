# ID mapping policy

Flow: **raw data → cleaned table → mapping table → graph/model input**. Original identifiers and source records are always retained in cleaned/audit tables.

- Disease: raw positive numeric OrphaCode → `ORPHA:<integer>`. ORPHA is the only canonical disease namespace in v1. OMIM/MONDO/UMLS/ICD are xrefs, never assumed 1:1 or merged by disease name. Without raw alignment evidence, external disease annotations remain audit-only.
- Gene: source-declared numeric HGNC xref → `HGNC:<integer>`; symbol and NCBI/Entrez IDs are attributes/xrefs. `resolved` means the source supplies a syntactically valid HGNC ID, not that a current registry has approved the symbol. No alias/symbol guessing occurs.
- Missing/ambiguous/invalid HGNC: `ORPHANET_LOCUS:<Gene/@id>` retained with canonical_status=unresolved and rankable=false. Missing both usable HGNC and source locus ID fails ingestion, rather than merging records by symbol. Fallback targets are absent from default split, negative target universe and Gene Top-K training targets.
- NCBI Gene-HPO: map only through a unique direct Orphanet external NCBI reference. Otherwise keep source records in gene_phenotype_unresolved.csv; a symbol coincidence is never sufficient. The current gene XML has no NCBI/Entrez xrefs, so a registry-backed mapping remains a blocker for canonical Gene-HPO edges.

## HPO

1. Active ID is retained.
2. Active term alt_id maps to its primary ID.
3. Obsolete ID with one replaced_by follows the replacement chain.
4. Consider-only or multiple replacements remain unresolved; no choice is guessed.
5. Unknown IDs and replacement cycles are audited unresolved.

The canonical catalog extends the team active-term parser. It separates OBO relationship IDs from comments/modifier blocks such as `{xref=...}`; the legacy runtime loader is unchanged. Invalid IS_A reference/cycle fails QC. hpo_id_original is preserved separately from hpo_id.

## Negative and frequency policy

NOT qualifier tokens (trimmed, case-normalized), Orphanet Excluded (0%), and explicit zero proportions are preserved as negative assertions and excluded from positive HAS_PHENOTYPE. No negative relation is emitted in v1; a future EXCLUDED_PHENOTYPE experiment can use the audit.

Canonical offline weights: 1.000 / 0.895 / 0.545 / 0.170 / 0.025 / 0.000 for Obligate / Very frequent / Frequent / Occasional / Very rare / Excluded. HPO frequency-category IDs map to the same values. Exact percentage/ratio inputs are parsed; unknown/missing remain NULL.

The shared offline constant/function is app/etl/frequency.py, consumed by canonical build and the existing offline frequency script. Existing user-facing runtime weights/defaults remain unchanged (0.90/0.75/0.40/0.15, unknown 0.30); this export does not claim that those policies are equivalent.
