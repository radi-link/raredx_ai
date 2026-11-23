# Canonical heterogeneous graph schema

Flow: **raw data → cleaned table → mapping table → graph/model input**.

Nodes: Phenotype=`HP:<7 digits>` active terms; Disease=`ORPHA:<integer>`; Gene=`HGNC:<integer>` or unresolved `ORPHANET_LOCUS:<source locus>`.

| Relation | Direction | Meaning |
|---|---|---|
| IS_A | Phenotype → Phenotype | child → parent ontology assertion |
| HAS_PHENOTYPE | Disease → Phenotype | positive source phenotype association |
| ASSOCIATED_WITH | Disease → Gene | raw Orphanet evidence aggregated by canonical pair |
| ASSOCIATED_PHENOTYPE | Gene → Phenotype | source Gene-HPO association with verified ID mapping; reference graph only |

Relation names and directions match the existing team Neo4j contract. Export tables are independent of the legacy Neo4j storage/query pipeline: this work does not automatically load a database or rewrite runtime ranking. Unresolved loci remain visible but not rankable. Disease nodes use the union, supporting phenotype-only, gene-only, and combined diseases; there is no inner join.

Full-reference graph edges must not be used blindly for training. `training_message_edges.csv` excludes heldout DG and all Gene-HPO edges. The latter can reveal a heldout disease/gene path through source disease provenance. Runtime gene symbol indexes and raw NCBI IDs are not a canonical graph consumer until a separately tested adapter is built.

A/B/C training exports use identical seed, canonical node universe and train pair list. A edge_weight=evidence_score; B edge_weight=evidence_score_idf with train-only fit; C intentionally omits edge_weight. Other evidence attributes remain audit fields: training consumers must select the declared ablation inputs and must not feed audit target/provenance fields as features. No model training or performance result is included.
