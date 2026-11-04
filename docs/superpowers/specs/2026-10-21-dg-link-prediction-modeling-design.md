# Disease–Gene Link Prediction Modeling — Design

Status: proposed
Author: fabxoe (via Claude Code interview)
Date: 2026-10-21

## Intent and why

`docs/data/modeling_handoff.md` (merged from `feature/data-knowledge-graph`)
hands off a ready-to-use knowledge graph and leakage-safe Disease–Gene (DG)
split, but states explicitly: *"모델 학습·성능 평가는 아직 수행하지 않았습니다"*
(model training/evaluation has not been done yet). This project is that
first pass: build a modeling module that loads the handed-off graph, trains
a baseline and a GNN to predict `(Disease, ASSOCIATED_WITH, Gene)` links,
and evaluates both with the ranking metrics the handoff doc anticipates
(Hits@1/10/50, MRR).

This is a research/offline pipeline, not a change to the runtime ranking
service. `app/graph/dataset.py`, `app/graph/splits.py`, `app/graph/qc.py`
(already merged) build the graph/splits from raw source; this project
consumes their *output*, not their code path.

## Goals

- A new `app/modeling/` module that trains and evaluates DG link prediction
  models reproducibly (seed 42) on the handed-off graph package.
- Two models in one pass: a small baseline (gene prior + bilinear) and a
  2-layer heterogeneous GNN (RGCN-style relational conv) + link decoder,
  built with PyTorch Geometric (PyG).
- Bidirectional ranking evaluation (corrupt Gene side holding Disease fixed,
  and corrupt Disease side holding Gene fixed) reporting Hits@1/10/50 and
  MRR per direction and combined, on validation and test splits.
- CPU-only training that finishes in a reasonable research-iteration time
  (this is a small graph: 30,358 nodes / 147,768 edges, 8,341 DG pairs).

## Non-goals

- Re-deriving the graph/splits from raw HPO/Orphanet source. We use the
  already-built `raredx-data-knowledge-graph-v1` package as-is.
- Gene–HPO edges, Neo4j loading, patient-case evaluation, or changes to the
  existing runtime ranking/Cypher pipeline — all explicitly out of scope
  per the handoff doc and the earlier Cursor task spec.
- Hyperparameter search infrastructure, experiment tracking integration, or
  a training-as-a-service API. This is a script-driven research pipeline.
- Learning-to-rank over the *production* candidate list — that is the
  existing `reranking` layer's job per `AGENTS.md`; this project produces
  an offline research model, not a replacement for it.

## Anti-goals (would make this a failure even if "it runs")

- Any leakage: val/test DG positives entering the message-passing graph,
  or val/test statistics being fit. The handoff's `split_policy.md`
  constraints must hold through our loader, not just through the upstream
  builder.
- Negative pairs reused across train/val/test (the pool is unsplit by
  design; we own the split and must not let it overlap).
- Using `graph_edges.csv` / `disease_gene_edges.csv` as model input — the
  handoff doc flags these as full-reference views containing heldout
  answers.

## Data (fixed inputs, from the team-provided package)

Source: `raredx-data-knowledge-graph-v1/graph/` (the package referenced in
`docs/data/modeling_handoff.md`). Files this project reads:

| File | Role |
|---|---|
| `hpo_nodes.csv`, `disease_nodes.csv`, `gene_nodes.csv` | node universe |
| `training_message_edges.csv` | message-passing graph (IS_A + positive Disease-HPO + train DG) |
| `splits/train.csv`, `splits/validation.csv`, `splits/test.csv` | target DG pairs |
| `negative_pairs.csv` | unsplit negative pool (8,339 pairs, `label=0`) |
| `splits/idf_fit.json` | train-only fit stats, if an ablation weight is used |

IDs stay strings (`ORPHA:<int>`, `HGNC:<int>`, `HP:<7 digits>`). Unresolved
genes (`ORPHANET_LOCUS:*`, `rankable=false`) are excluded, consistent with
the handoff's default target exclusion.

The raw package (`raredx-data-knowledge-graph-v1/graph/`) has been copied
into `data/processed/graph/` (matches the path `scripts/build_processed.py`
already writes to; excluded from git via the existing `data/processed/*`
`.gitignore` rule). The loader reads from this fixed, repo-relative path.

## How it gets made (decided)

- **Module location:** `app/modeling/` — follows the existing `app/`
  layout from `AGENTS.md` (`app/graph`, `app/etl`, …) rather than the
  unused `src/train/` path from the earlier Cursor plan. Files:
  - `app/modeling/loader.py` — CSV → PyG `HeteroData` (node ID maps per
    type, one edge type per relation, negative pool split).
  - `app/modeling/models.py` — baseline (gene prior + bilinear scorer) and
    the GNN (2-layer `RGCNConv`/`HeteroConv` + bilinear or DistMult-style
    decoder).
  - `app/modeling/train.py` — training loop for both models, CPU, seed 42.
  - `app/modeling/evaluate.py` — bidirectional filtered ranking (Hits@K,
    MRR) on validation/test.
  - `scripts/train_dg_link_model.py` — thin CLI entry point, mirroring the
    existing `scripts/build_processed.py` / `scripts/build_gene_mapping.py`
    pattern.
- **Framework:** PyTorch Geometric. Not in `AGENTS.md`'s preferred-library
  list (which only names PyTorch), but it's the de facto standard for
  heterogeneous GNNs and the graph is small enough that CPU-only PyG is
  fine. Added as a project dependency via `uv add torch-geometric`. If the
  compiled `torch-scatter`/`torch-sparse` extensions fail to install on
  this machine, fall back to PyG's pure-PyTorch message-passing backend
  (no separate extension packages needed on recent PyG versions) — your
  call if that fallback is hit, otherwise no visible difference.
- **Negative sampling:**
  - *Training:* `negative_pairs.csv` (8,339 rows) is shuffled with seed 42
    and split into train/val/test using the same sizes as the positive
    splits (6,673 / 833 / 833), mirroring the method in
    `docs/data/split_policy.md` (seeded shuffle, `floor(N*fraction)`
    holdouts, remainder to train). This gives the BCE loss a fixed,
    reproducible, non-overlapping negative set per split.
  - *Evaluation:* independent of `negative_pairs.csv`. Hits@K/MRR use
    **filtered ranking**: for each test/val positive pair, corrupt one
    side (Gene or Disease) against the node universe of that type
    *restricted to resolved, `rankable=true` nodes* (unresolved genes are
    never valid answers per the handoff's own target-exclusion policy, so
    including them as corruption candidates would inflate Hits@K/MRR),
    remove all other known positive pairs from the candidate pool (the
    standard KG-eval "filtered" setting), and rank the true tail/head
    among the rest. Done once with Gene corrupted, once with Disease
    corrupted, reported separately and combined.
- **Message-passing direction:** `training_message_edges.csv` stores each
  relation one-directional (IS_A child→parent, Disease→HPO, Disease→Gene).
  The loader adds the reverse relation type for each of these when
  building the PyG `HeteroData` (e.g. via `T.ToUndirected()` or explicit
  `rev_IS_A` / `rev_HAS_PHENOTYPE` / `rev_ASSOCIATED_WITH` edge types), so
  the RGCN layers can pass information both ways. Without this the GNN
  would have no structural advantage over the bilinear baseline — this is
  a correctness requirement, not a tunable.
- **Cold-start genes (no train-graph edges):** because Gene–HPO edges are
  universally absent from this graph, a gene's only possible source of
  message-passing signal is `ASSOCIATED_WITH` edges that landed in the
  *train* split. A gene whose sole known DG association is in val/test has
  no edges in the message-passing graph and never receives a gradient
  update — its embedding stays at random init. Evaluation reports these
  separately: compute Hits@K/MRR once over the full val/test set, and
  again restricted to the subset of pairs whose Gene (or Disease, for the
  reverse direction) has at least one train-graph edge, labeling the
  excluded group as a known cold-start population rather than averaging
  them in silently. No mitigation (e.g. fallback embeddings) in this v1 —
  revisit only if the cold-start group turns out to be large enough to
  matter.
- **Model sizing:** small by construction — 8,339 labeled pairs rules out
  anything large. Baseline first (sanity check the pipeline), then GNN,
  both reported side by side in the same run's output.
- **Scope:** baseline + GNN in one pass, not baseline-only.

## What's left to the builder

- Exact bilinear/DistMult decoder formula and embedding dimension
  (pick something small, e.g. 64–128 dim, given 8,339 pairs — avoid
  overfitting; document the choice).
- RGCN layer width/dropout/optimizer/learning rate/epoch count/early
  stopping metric (suggest: early-stop on validation MRR).
- Exact output format for metrics (suggest: a JSON report under
  `reports/` or `docs/data/`, following the repo's existing
  `qc_report.json`-style convention) and whether training logs anything
  beyond stdout.
- Whether `idf_fit.json` / ablation-weighted edges (A/B/C files) are used
  as an edge feature in v1 or deferred to a later iteration (suggest:
  deferred — out of the agreed scope above, revisit after a working
  unweighted baseline).
- Test coverage shape: given `AGENTS.md` requires testable components,
  the builder should add at least loader correctness tests (no leakage,
  negative split sizes/non-overlap, ID map round-trips) following the
  existing `tests/test_split_leakage.py` style.

## Decisions log

- Data package location: `data/processed/graph/` (confirmed).
- Cold-start genes/diseases with no train-graph edges: report separately
  rather than averaging them into the main Hits@K/MRR numbers, with no
  mitigation built in this v1 (confirmed).
