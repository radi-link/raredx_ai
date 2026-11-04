# Disease–Gene Link Prediction Modeling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `app/modeling/` so it loads the handed-off Disease–Gene knowledge graph package, trains a baseline and a heterogeneous GNN to predict `(Disease, ASSOCIATED_WITH, Gene)` links, and evaluates both with bidirectional filtered Hits@1/10/50 and MRR.

**Architecture:** A loader turns the CSV package in `data/processed/graph/` into a PyG `HeteroData` object (with reverse edges per relation) plus split pair/negative lists. Two models — a no-graph baseline (gene prior + bilinear) and a 2-layer RGCN + bilinear decoder — train on the same positive/negative pairs via BCE. A shared evaluator does filtered, bidirectional ranking and reports cold-start pairs separately. A CLI script wires it together and writes a JSON report.

**Tech Stack:** Python 3.12, PyTorch (already a transitive dependency via `sentence-transformers`), PyTorch Geometric (new dependency), pytest.

**Spec:** `docs/superpowers/specs/2026-10-21-dg-link-prediction-modeling-design.md`

## Global Constraints

- Python 3.12, type hints required, modular architecture, testable components (per `AGENTS.md`).
- Seed 42 everywhere splits or randomness occur (negative split, model init, training).
- CPU only — no code may assume CUDA availability.
- Model location: `app/modeling/` (not `src/train/`).
- Data package location: `data/processed/graph/` (already copied there).
- Framework: PyTorch Geometric (`torch-geometric`), added via `uv add torch-geometric`.
- Never read `graph_edges.csv` or `disease_gene_edges.csv` as model input — reference-only, contains heldout answers.
- Message-passing graph = `training_message_edges.csv` only, plus its reverse relations added by the loader. Val/test DG positives must never enter it.
- Ranking candidate pool for filtered evaluation = resolved, `rankable=true` nodes of the corrupted type only.
- Cold-start pairs (gene or disease with zero train-graph edges) are reported as a separate metrics block, never silently averaged into the main numbers.

## Review Focus

- Val/test DG positives leaking into the message-passing `HeteroData` the GNN trains on (the spec's core leakage boundary) — expect the loader to reject or simply never include them, and a test to prove it against the real `training_message_edges.csv` contract.
- Negative pairs reused across train/val/test, or overlapping a known positive — `split_negative_pairs` must produce disjoint sets sized like the positive splits, every time, for seed 42.
- Non-rankable / unresolved genes appearing as true answers or as corruption candidates in evaluation — both are domain errors the spec calls out explicitly.
- A pair whose Gene (or Disease) has zero edges in the message-passing graph being counted in the main Hits@K/MRR numbers instead of the cold-start block.
- Reverse edges silently missing from the GNN's `HeteroData`, which would make the GNN run without error but perform no better than the baseline — a correctness bug that looks like a modeling result.

---

## File Structure

- `app/modeling/__init__.py` — empty, marks the package.
- `app/modeling/loader.py` — CSV → in-memory graph package (node indices, `HeteroData`, split pairs, split negatives).
- `app/modeling/models.py` — `BaselineScorer` and `GNNLinkPredictor`.
- `app/modeling/train.py` — training loops shared by both models.
- `app/modeling/evaluate.py` — filtered bidirectional ranking metrics.
- `scripts/train_dg_link_model.py` — CLI entry point producing a JSON report.
- `tests/fixtures/modeling/` — tiny synthetic CSVs mirroring the real schema (new, this plan).
- `tests/test_modeling_loader.py`, `tests/test_modeling_models.py`, `tests/test_modeling_train.py`, `tests/test_modeling_evaluate.py`, `tests/test_train_dg_link_model_cli.py` — new tests, one file per module above.

---

### Task 1: Fixtures and CSV loading primitives

**Files:**
- Create: `tests/fixtures/modeling/gene_nodes.csv`, `disease_nodes.csv`, `hpo_nodes.csv`, `training_message_edges.csv`, `splits/train.csv`, `splits/validation.csv`, `splits/test.csv`, `negative_pairs.csv`
- Create: `app/modeling/__init__.py`
- Create: `app/modeling/loader.py`
- Test: `tests/test_modeling_loader.py`

**Interfaces:**
- Produces:
  - `NodeRecord = dict[str, Any]`
  - `load_node_csv(path: Path) -> dict[str, NodeRecord]` — keyed by `node_id`, includes `node_type`, `rankable` (bool, default `True` if column absent), `canonical_status`.
  - `load_message_edge_csv(path: Path) -> list[dict[str, Any]]` — raw rows (`source_id`, `target_id`, `relation`, ...).
  - `load_pair_csv(path: Path) -> list[tuple[str, str]]` — `(source_id, target_id)` for rows where `relation == "ASSOCIATED_WITH"`; raises `ValueError` on any other relation present.
  - `load_negative_pairs_csv(path: Path) -> list[tuple[str, str]]` — `(source_id, target_id)` for rows where `label == "0"`.

**Fixture design (keep tiny, deterministic, schema-faithful):**
- 4 `Disease` nodes (`ORPHA:1`..`ORPHA:4`), 5 `Gene` nodes (`HGNC:1`..`HGNC:5`, one with `rankable=false`), 3 `Phenotype` nodes (`HP:0000001`..`HP:0000003`).
- `training_message_edges.csv`: a few `IS_A` (Phenotype→Phenotype), `HAS_PHENOTYPE` (Disease→Phenotype), and `ASSOCIATED_WITH` (Disease→Gene, train split only) rows, using the real column set (header copied from the real file, blank cells where not applicable).
- `splits/train.csv` (4 pairs), `splits/validation.csv` (1 pair), `splits/test.csv` (1 pair) — real column set, `split`/`seed`/`split_version` populated.
- `negative_pairs.csv`: 6 rows (`source_id,target_id,label`), disjoint from all positive pairs above.
- Include one gene (`HGNC:5`) whose only `ASSOCIATED_WITH` edge is in `splits/test.csv` and absent from `training_message_edges.csv` — this is the fixture's cold-start case, reused by Task 4 and Task 8.

- [ ] **Step 1: Write the fixture CSVs listed above** (plain file creation, matching the real header rows from `data/processed/graph/*.csv` and `data/processed/graph/splits/*.csv`).

- [ ] **Step 2: Write failing tests for the four loader functions**

```python
def test_load_node_csv_indexes_by_node_id_and_parses_rankable():
    nodes = load_node_csv(FIXTURE_DIR / "gene_nodes.csv")
    assert nodes["HGNC:1"]["node_type"] == "Gene"
    assert nodes["HGNC:5"]["rankable"] is False

def test_load_message_edge_csv_returns_all_relations():
    edges = load_message_edge_csv(FIXTURE_DIR / "training_message_edges.csv")
    relations = {e["relation"] for e in edges}
    assert relations == {"IS_A", "HAS_PHENOTYPE", "ASSOCIATED_WITH"}

def test_load_pair_csv_rejects_non_associated_with_rows(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("source_id,target_id,relation\nORPHA:1,HP:0000001,HAS_PHENOTYPE\n")
    with pytest.raises(ValueError):
        load_pair_csv(bad)

def test_load_negative_pairs_csv_returns_label_zero_pairs():
    negatives = load_negative_pairs_csv(FIXTURE_DIR / "negative_pairs.csv")
    assert len(negatives) == 6
    assert all(isinstance(p, tuple) and len(p) == 2 for p in negatives)
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_modeling_loader.py -v`
Expected: FAIL (`ModuleNotFoundError` / `ImportError` for `app.modeling.loader`)

- [ ] **Step 4: Implement the four functions in `app/modeling/loader.py`**

Pure `csv.DictReader`-based parsing, no pandas (match `app/graph/dataset.py`'s stdlib-`csv` convention). `rankable` parses the string `"true"`/`"false"` (case-insensitive) to `bool`, defaulting to `True` when the column is missing or empty, per the repo's existing boolean-cell convention in `docs/data/data_dictionary.md`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_modeling_loader.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add app/modeling/__init__.py app/modeling/loader.py tests/fixtures/modeling tests/test_modeling_loader.py
git commit -m "feat(modeling): add CSV loading primitives and fixtures"
```

---

### Task 2: Negative pool split

**Files:**
- Modify: `app/modeling/loader.py`
- Test: `tests/test_modeling_loader.py`

**Interfaces:**
- Consumes: `load_negative_pairs_csv` (Task 1).
- Produces: `split_negative_pairs(negatives: list[tuple[str, str]], *, split_sizes: dict[str, int], seed: int = 42) -> dict[str, list[tuple[str, str]]]` — keys `"train"`, `"validation"`, `"test"`; raises `ValueError` if `sum(split_sizes.values()) > len(negatives)`.

- [ ] **Step 1: Write failing tests**

```python
def test_split_negative_pairs_matches_requested_sizes_and_is_disjoint():
    negatives = load_negative_pairs_csv(FIXTURE_DIR / "negative_pairs.csv")
    result = split_negative_pairs(negatives, split_sizes={"train": 4, "validation": 1, "test": 1})
    assert [len(result[name]) for name in ("train", "validation", "test")] == [4, 1, 1]
    seen = [p for group in result.values() for p in group]
    assert len(seen) == len(set(seen))

def test_split_negative_pairs_is_deterministic_for_fixed_seed():
    negatives = load_negative_pairs_csv(FIXTURE_DIR / "negative_pairs.csv")
    first = split_negative_pairs(negatives, split_sizes={"train": 4, "validation": 1, "test": 1}, seed=42)
    second = split_negative_pairs(list(reversed(negatives)), split_sizes={"train": 4, "validation": 1, "test": 1}, seed=42)
    assert first == second

def test_split_negative_pairs_rejects_oversized_request():
    negatives = load_negative_pairs_csv(FIXTURE_DIR / "negative_pairs.csv")
    with pytest.raises(ValueError):
        split_negative_pairs(negatives, split_sizes={"train": 10, "validation": 1, "test": 1})
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_modeling_loader.py -k negative -v`
Expected: FAIL (function not defined)

- [ ] **Step 3: Implement `split_negative_pairs`**

Sort the input pairs for a stable base order, shuffle with `random.Random(seed)` (mirrors `app/graph/splits.py`'s local-RNG convention — never touch the global `random` module), then slice sequentially into `train`/`validation`/`test` using the exact requested sizes (order of keys: train, then validation, then test).

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_modeling_loader.py -k negative -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/modeling/loader.py tests/test_modeling_loader.py
git commit -m "feat(modeling): split negative pool to match positive split sizes"
```

---

### Task 3: HeteroData builder with reverse edges

**Files:**
- Modify: `pyproject.toml` (add `torch-geometric` dependency)
- Modify: `app/modeling/loader.py`
- Test: `tests/test_modeling_loader.py`

**Interfaces:**
- Consumes: `load_node_csv`, `load_message_edge_csv` (Task 1).
- Produces:
  - `@dataclass class NodeIndex: ids: list[str]; index: dict[str, int]; rankable: set[str]` — `rankable` holds the subset of `ids` with `rankable is True`.
  - `build_node_index(nodes: dict[str, NodeRecord]) -> NodeIndex`
  - `build_hetero_data(node_indexes: dict[str, NodeIndex], message_edges: list[dict[str, Any]]) -> "torch_geometric.data.HeteroData"` — one edge type per relation in `message_edges` keyed as `(source_node_type, relation, target_node_type)`, plus a reverse edge type `(target_node_type, f"rev_{relation}", source_node_type)` for every relation, built with `torch_geometric.transforms.ToUndirected` semantics (explicit reverse edge type, not merged into the forward type, so direction stays inspectable in tests).

- [ ] **Step 1: Add the dependency**

Run: `uv add torch-geometric`
Expected: resolves and installs on top of the existing CPU-only `torch`. If `torch-scatter`/`torch-sparse` wheels fail for this platform, fall back to `uv add torch-geometric --no-deps` plus whatever minimal extra PyG needs at import time for `HeteroData`/`RGCNConv` on its pure-PyTorch backend (verify with Step 2 below before deciding this is necessary).

- [ ] **Step 2: Run `uv run python -c "import torch_geometric; from torch_geometric.data import HeteroData; from torch_geometric.nn import RGCNConv"` to confirm the install is usable**

Expected: no ImportError.

- [ ] **Step 3: Write failing tests**

```python
def test_build_node_index_separates_rankable_ids():
    nodes = load_node_csv(FIXTURE_DIR / "gene_nodes.csv")
    index = build_node_index(nodes)
    assert "HGNC:5" in index.ids and "HGNC:5" not in index.rankable

def test_build_hetero_data_has_forward_and_reverse_edge_types():
    genes = build_node_index(load_node_csv(FIXTURE_DIR / "gene_nodes.csv"))
    diseases = build_node_index(load_node_csv(FIXTURE_DIR / "disease_nodes.csv"))
    phenotypes = build_node_index(load_node_csv(FIXTURE_DIR / "hpo_nodes.csv"))
    edges = load_message_edge_csv(FIXTURE_DIR / "training_message_edges.csv")
    data = build_hetero_data(
        {"Gene": genes, "Disease": diseases, "Phenotype": phenotypes}, edges
    )
    assert ("Disease", "ASSOCIATED_WITH", "Gene") in data.edge_types
    assert ("Gene", "rev_ASSOCIATED_WITH", "Disease") in data.edge_types
    forward = data["Disease", "ASSOCIATED_WITH", "Gene"].edge_index
    reverse = data["Gene", "rev_ASSOCIATED_WITH", "Disease"].edge_index
    assert forward.shape[1] == reverse.shape[1]
    assert torch.equal(forward[[0, 1]], reverse[[1, 0]])

def test_build_hetero_data_excludes_test_split_pairs():
    # training_message_edges.csv fixture must not contain the ORPHA:4/HGNC:5 test pair
    edges = load_message_edge_csv(FIXTURE_DIR / "training_message_edges.csv")
    test_pairs = set(load_pair_csv(FIXTURE_DIR / "splits/test.csv"))
    message_pairs = {
        (e["source_id"], e["target_id"]) for e in edges if e["relation"] == "ASSOCIATED_WITH"
    }
    assert not (message_pairs & test_pairs)
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `uv run pytest tests/test_modeling_loader.py -k "node_index or hetero" -v`
Expected: FAIL

- [ ] **Step 5: Implement `NodeIndex`, `build_node_index`, `build_hetero_data`**

`build_hetero_data` groups `message_edges` by `relation`, looks up each relation's `(source_node_type, target_node_type)` from the node indexes (infer source/target type from which `NodeIndex` contains the id — `Disease`/`Gene`/`Phenotype` id prefixes `ORPHA:`/`HGNC:`/`HP:` already disambiguate this), builds a `torch.tensor` edge_index per type, and adds the mirrored reverse tensor under `rev_{relation}`.

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/test_modeling_loader.py -v`
Expected: PASS (all loader tests so far)

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml uv.lock app/modeling/loader.py tests/test_modeling_loader.py
git commit -m "feat(modeling): build heterogeneous PyG graph with reverse edges"
```

---

### Task 4: Graph package assembly

**Files:**
- Modify: `app/modeling/loader.py`
- Test: `tests/test_modeling_loader.py`

**Interfaces:**
- Consumes: everything from Tasks 1–3.
- Produces:
  - `@dataclass class GraphPackage: data: HeteroData; node_indexes: dict[str, NodeIndex]; positives: dict[str, list[tuple[str, str]]]; negatives: dict[str, list[tuple[str, str]]]` — `positives`/`negatives` keyed `"train"`/`"validation"`/`"test"`.
  - `load_graph_package(graph_dir: Path, *, seed: int = 42) -> GraphPackage` — reads `gene_nodes.csv`, `disease_nodes.csv`, `hpo_nodes.csv`, `training_message_edges.csv`, `splits/train.csv`, `splits/validation.csv`, `splits/test.csv`, `negative_pairs.csv` from `graph_dir`; calls `split_negative_pairs` with `split_sizes` equal to the three positive splits' lengths.

- [ ] **Step 1: Write failing test**

```python
def test_load_graph_package_end_to_end_from_fixtures():
    package = load_graph_package(FIXTURE_DIR, seed=42)
    assert set(package.positives) == {"train", "validation", "test"}
    assert len(package.negatives["train"]) == len(package.positives["train"])
    assert ("Disease", "ASSOCIATED_WITH", "Gene") in package.data.edge_types
    train_pairs = set(package.positives["train"])
    val_test_pairs = set(package.positives["validation"]) | set(package.positives["test"])
    assert not (train_pairs & val_test_pairs)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_modeling_loader.py -k end_to_end -v`
Expected: FAIL

- [ ] **Step 3: Implement `GraphPackage` and `load_graph_package`**

Thin composition of the Task 1–3 functions; no new parsing logic.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_modeling_loader.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/modeling/loader.py tests/test_modeling_loader.py
git commit -m "feat(modeling): assemble full graph package from CSV directory"
```

---

### Task 5: Baseline model

**Files:**
- Create: `app/modeling/models.py`
- Test: `tests/test_modeling_models.py`

**Interfaces:**
- Consumes: `GraphPackage.node_indexes["Gene"].ids`, `GraphPackage.positives["train"]` (for gene prior counts).
- Produces:
  - `compute_gene_prior(train_positives: list[tuple[str, str]], gene_ids: list[str]) -> dict[str, float]` — normalized train-positive frequency per gene id (0.0 for unseen genes).
  - `class BaselineScorer(torch.nn.Module)`:
    - `__init__(self, disease_ids: list[str], gene_ids: list[str], gene_prior: dict[str, float], *, embedding_dim: int = 32, seed: int = 42)`
    - `forward(self, disease_idx: torch.LongTensor, gene_idx: torch.LongTensor) -> torch.Tensor` — returns a 1-D logit tensor, one per input pair: `gene_prior_logit[gene] + disease_emb @ W @ gene_emb`.

- [ ] **Step 1: Write failing tests**

```python
def test_compute_gene_prior_normalizes_and_defaults_to_zero():
    prior = compute_gene_prior([("ORPHA:1", "HGNC:1"), ("ORPHA:2", "HGNC:1")], ["HGNC:1", "HGNC:2"])
    assert prior["HGNC:1"] == 1.0
    assert prior["HGNC:2"] == 0.0

def test_baseline_scorer_forward_returns_one_logit_per_pair():
    torch.manual_seed(0)
    model = BaselineScorer(["ORPHA:1", "ORPHA:2"], ["HGNC:1", "HGNC:2"], {"HGNC:1": 1.0, "HGNC:2": 0.0})
    scores = model(torch.tensor([0, 1]), torch.tensor([1, 0]))
    assert scores.shape == (2,)
    assert torch.isfinite(scores).all()

def test_baseline_scorer_is_deterministic_for_fixed_seed():
    model_a = BaselineScorer(["ORPHA:1"], ["HGNC:1"], {"HGNC:1": 0.5}, seed=42)
    model_b = BaselineScorer(["ORPHA:1"], ["HGNC:1"], {"HGNC:1": 0.5}, seed=42)
    assert torch.equal(model_a.disease_embedding.weight, model_b.disease_embedding.weight)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_modeling_models.py -k baseline -v`
Expected: FAIL

- [ ] **Step 3: Implement `compute_gene_prior` and `BaselineScorer`**

`compute_gene_prior` divides each gene's train-positive count by the max count (so the most-associated gene gets 1.0); store as a registered buffer indexed by the model's internal gene order. `seed` seeds a local `torch.Generator` passed to `nn.init` calls — do not touch global torch RNG state (mirrors the repo's local-RNG convention from `app/graph/splits.py`).

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_modeling_models.py -k baseline -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/modeling/models.py tests/test_modeling_models.py
git commit -m "feat(modeling): add gene-prior + bilinear baseline model"
```

---

### Task 6: GNN model

**Files:**
- Modify: `app/modeling/models.py`
- Test: `tests/test_modeling_models.py`

**Interfaces:**
- Consumes: `GraphPackage.data` (Task 3/4), node type names `"Disease"`, `"Gene"`, `"Phenotype"`.
- Produces:
  - `class GNNLinkPredictor(torch.nn.Module)`:
    - `__init__(self, metadata: tuple[list[str], list[tuple[str, str, str]]], *, hidden_dim: int = 64, embedding_dim: int = 64, seed: int = 42)` — `metadata` is `HeteroData.metadata()`; builds per-node-type input embeddings plus two `HeteroConv` layers (one `RGCNConv`-equivalent relation-specific linear per edge type, as PyG's `HeteroConv({edge_type: SAGEConv(...) or GraphConv(...)}, aggr="sum")` pattern) producing `Disease` and `Gene` output embeddings.
    - `encode(self, data: HeteroData) -> dict[str, torch.Tensor]` — returns `{"Disease": tensor[num_disease, embedding_dim], "Gene": tensor[num_gene, embedding_dim]}`.
    - `decode(self, disease_emb: torch.Tensor, gene_emb: torch.Tensor, disease_idx: torch.LongTensor, gene_idx: torch.LongTensor) -> torch.Tensor` — bilinear decoder, same shape contract as `BaselineScorer.forward`.

- [ ] **Step 1: Write failing tests**

```python
def test_gnn_encode_returns_embedding_per_node():
    package = load_graph_package(FIXTURE_DIR, seed=42)
    model = GNNLinkPredictor(package.data.metadata(), hidden_dim=8, embedding_dim=8, seed=42)
    embeddings = model.encode(package.data)
    assert embeddings["Disease"].shape == (len(package.node_indexes["Disease"].ids), 8)
    assert embeddings["Gene"].shape == (len(package.node_indexes["Gene"].ids), 8)

def test_gnn_decode_returns_one_logit_per_pair():
    package = load_graph_package(FIXTURE_DIR, seed=42)
    model = GNNLinkPredictor(package.data.metadata(), hidden_dim=8, embedding_dim=8, seed=42)
    embeddings = model.encode(package.data)
    scores = model.decode(embeddings["Disease"], embeddings["Gene"], torch.tensor([0]), torch.tensor([0]))
    assert scores.shape == (1,)
    assert torch.isfinite(scores).all()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_modeling_models.py -k gnn -v`
Expected: FAIL

- [ ] **Step 3: Implement `GNNLinkPredictor`**

Two stacked `HeteroConv` layers over all edge types present in `metadata` (forward and `rev_*`), ReLU between layers, final linear projection to `embedding_dim` per node type. This is the one place the implementer picks the exact PyG layer class (`SAGEConv`/`GraphConv`/`GATConv` are all acceptable relation-specific convs for `HeteroConv`) and dropout — note the choice in a short comment.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_modeling_models.py -v`
Expected: PASS (baseline and GNN tests)

- [ ] **Step 5: Commit**

```bash
git add app/modeling/models.py tests/test_modeling_models.py
git commit -m "feat(modeling): add 2-layer heterogeneous GNN link predictor"
```

---

### Task 7: Training loop

**Files:**
- Create: `app/modeling/train.py`
- Test: `tests/test_modeling_train.py`

**Interfaces:**
- Consumes: `GraphPackage` (Task 4), `BaselineScorer` / `GNNLinkPredictor` (Tasks 5–6).
- Produces:
  - `@dataclass class TrainingHistory: train_loss: list[float]; validation_loss: list[float]`
  - `train_baseline(package: GraphPackage, *, epochs: int = 50, lr: float = 0.01, seed: int = 42) -> tuple[BaselineScorer, TrainingHistory]`
  - `train_gnn(package: GraphPackage, *, epochs: int = 50, lr: float = 0.01, hidden_dim: int = 64, embedding_dim: int = 64, seed: int = 42) -> tuple[GNNLinkPredictor, TrainingHistory]`

Both functions build the pair tensors from `package.positives["train"]` (label 1) and `package.negatives["train"]` (label 0), optimize `BCEWithLogitsLoss` with `torch.optim.Adam`, and at each epoch also compute (not backprop) the same loss over `validation` positives/negatives into `TrainingHistory.validation_loss`.

- [ ] **Step 1: Write failing tests**

```python
def test_train_baseline_reduces_loss_on_fixture_graph():
    package = load_graph_package(FIXTURE_DIR, seed=42)
    _, history = train_baseline(package, epochs=20, seed=42)
    assert history.train_loss[-1] < history.train_loss[0]

def test_train_baseline_is_deterministic_for_fixed_seed():
    package = load_graph_package(FIXTURE_DIR, seed=42)
    _, history_a = train_baseline(package, epochs=5, seed=42)
    _, history_b = train_baseline(package, epochs=5, seed=42)
    assert history_a.train_loss == history_b.train_loss

def test_train_gnn_reduces_loss_on_fixture_graph():
    package = load_graph_package(FIXTURE_DIR, seed=42)
    _, history = train_gnn(package, epochs=20, hidden_dim=8, embedding_dim=8, seed=42)
    assert history.train_loss[-1] < history.train_loss[0]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_modeling_train.py -v`
Expected: FAIL

- [ ] **Step 3: Implement `TrainingHistory`, `train_baseline`, `train_gnn`**

Seed `torch.manual_seed(seed)` locally at the start of each function (acceptable here — unlike `app/graph/splits.py`'s pair-sampling code, there is no caller-shared global RNG state to protect across a training run) before constructing the model, so model init and any dropout are reproducible.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_modeling_train.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/modeling/train.py tests/test_modeling_train.py
git commit -m "feat(modeling): add training loops for baseline and GNN"
```

---

### Task 8: Filtered bidirectional evaluator

**Files:**
- Create: `app/modeling/evaluate.py`
- Test: `tests/test_modeling_evaluate.py`

**Interfaces:**
- Consumes: `GraphPackage` (Task 4), a scoring callable matching `BaselineScorer.forward` / `GNNLinkPredictor.decode`'s pair-logit contract.
- Produces:
  - `@dataclass class RankingMetrics: hits_at_1: float; hits_at_10: float; hits_at_50: float; mrr: float; count: int`
  - `@dataclass class EvaluationReport: gene_corruption: RankingMetrics; disease_corruption: RankingMetrics; combined: RankingMetrics; cold_start_gene_corruption: RankingMetrics; cold_start_disease_corruption: RankingMetrics`
  - `evaluate_link_prediction(score_pair: Callable[[list[tuple[str, str]]], list[float]], package: GraphPackage, split: str) -> EvaluationReport` — `score_pair` takes a list of `(disease_id, gene_id)` string pairs and returns one score per pair (callers adapt `BaselineScorer`/`GNNLinkPredictor` to this signature outside this module, keeping `evaluate.py` model-agnostic).

**Cold-start determination:** a positive pair is in the cold-start set for gene-corruption if its gene has zero `ASSOCIATED_WITH` edges in `package.data["Disease", "ASSOCIATED_WITH", "Gene"]`'s train-graph edge list; symmetric for disease-corruption.

- [ ] **Step 1: Write failing tests**

```python
def test_evaluate_link_prediction_ranks_known_pair_first():
    # score_pair returns a high score for the true pair and low scores for every
    # candidate — expect Hits@1 == 1.0 and MRR == 1.0 for that split.
    package = load_graph_package(FIXTURE_DIR, seed=42)

    def perfect_scorer(pairs):
        true_pairs = set(package.positives["validation"])
        return [10.0 if pair in true_pairs else -10.0 for pair in pairs]

    report = evaluate_link_prediction(perfect_scorer, package, "validation")
    assert report.combined.hits_at_1 == 1.0
    assert report.combined.mrr == 1.0

def test_evaluate_link_prediction_separates_cold_start_pairs():
    package = load_graph_package(FIXTURE_DIR, seed=42)

    def constant_scorer(pairs):
        return [0.0 for _ in pairs]

    report = evaluate_link_prediction(constant_scorer, package, "test")
    # fixture's ORPHA:4/HGNC:5 test pair has no train-graph edge for HGNC:5
    assert report.cold_start_gene_corruption.count >= 1
    assert report.cold_start_gene_corruption.count + report.gene_corruption.count == len(package.positives["test"])

def test_evaluate_link_prediction_excludes_non_rankable_candidates():
    package = load_graph_package(FIXTURE_DIR, seed=42)
    rankable_genes = package.node_indexes["Gene"].rankable
    assert "HGNC:5" not in rankable_genes  # fixture's non-rankable gene must never be a candidate
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_modeling_evaluate.py -v`
Expected: FAIL

- [ ] **Step 3: Implement `RankingMetrics`, `EvaluationReport`, `evaluate_link_prediction`**

For each positive pair and each corruption direction: build the candidate list (true pair + every other rankable id of the corrupted type, minus ids that form a known positive pair with the fixed side, per the filtered-ranking rule in the spec), call `score_pair` once per direction in batch, rank by descending score, record the true pair's 1-indexed rank, and fold into `RankingMetrics` (`hits_at_k = mean(rank <= k)`, `mrr = mean(1 / rank)`). Route each pair into the cold-start or main bucket per the rule above before aggregating; `combined` merges `gene_corruption` and `disease_corruption` (excluding cold-start) into one `RankingMetrics`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_modeling_evaluate.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/modeling/evaluate.py tests/test_modeling_evaluate.py
git commit -m "feat(modeling): add filtered bidirectional ranking evaluator"
```

---

### Task 9: CLI entry point and JSON report

**Files:**
- Create: `scripts/train_dg_link_model.py`
- Test: `tests/test_train_dg_link_model_cli.py`

**Interfaces:**
- Consumes: `load_graph_package` (Task 4), `train_baseline`/`train_gnn` (Task 7), `evaluate_link_prediction` (Task 8).
- Produces: a `main(argv: list[str] | None = None) -> int` CLI function (mirrors `scripts/build_processed.py`'s `--canonical-graph --seed 42` flag style) accepting `--graph-dir` (default `data/processed/graph`), `--seed` (default `42`), `--epochs` (default `50`), `--output` (default `reports/dg_link_prediction_report.json`). Writes a JSON report with top-level keys `"baseline"` and `"gnn"`, each holding `{"validation": EvaluationReport-as-dict, "test": EvaluationReport-as-dict}`.

- [ ] **Step 1: Write failing test**

```python
def test_cli_writes_report_with_expected_keys(tmp_path):
    output = tmp_path / "report.json"
    exit_code = main([
        "--graph-dir", str(FIXTURE_DIR),
        "--seed", "42",
        "--epochs", "3",
        "--output", str(output),
    ])
    assert exit_code == 0
    report = json.loads(output.read_text())
    assert set(report) == {"baseline", "gnn"}
    assert set(report["baseline"]) == {"validation", "test"}
    assert "hits_at_1" in report["baseline"]["test"]["combined"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_train_dg_link_model_cli.py -v`
Expected: FAIL (script does not exist)

- [ ] **Step 3: Implement `scripts/train_dg_link_model.py`**

`argparse`-based CLI; loads the package once, trains both models, evaluates both on validation and test, serializes each `EvaluationReport` via `dataclasses.asdict`, writes indented JSON, creates the output directory if missing, returns `0`. Add a `reports/` directory with a `.gitkeep` if it doesn't already exist (check first — `data/processed/*` is gitignored, `reports/` is not, so this is new output the repo should track the directory for, not the generated report itself; add `reports/*.json` to `.gitignore` to keep generated reports out of git).

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_train_dg_link_model_cli.py -v`
Expected: PASS

- [ ] **Step 5: Run the full test suite**

Run: `uv run pytest -q`
Expected: all tests pass, including the pre-existing 177.

- [ ] **Step 6: Commit**

```bash
git add scripts/train_dg_link_model.py tests/test_train_dg_link_model_cli.py .gitignore
git commit -m "feat(modeling): add CLI to train and evaluate DG link prediction models"
```

---

## Self-Review Notes

- **Spec coverage:** data location (Task 4/9 default path), negative split policy (Task 2), reverse-edge message passing (Task 3), cold-start reporting (Task 8), rankable-only filtered candidates (Task 8), baseline+GNN scope (Tasks 5–6), bidirectional Hits@K/MRR (Task 8), CPU-only PyG (Task 3 step 1–2) — all covered.
- **What's-left-to-the-builder items from the spec** (embedding dim, decoder formula, RGCN width/dropout/optimizer/epochs/early-stopping, report location, idf/ablation deferral, loader leakage tests) are resolved as defaults inside task steps (e.g. `embedding_dim=64`, Adam, no early stopping in v1 — epochs is a CLI flag instead, which is a simpler v1 choice than tracking a stopping metric) rather than left open in this plan, consistent with "builder decides, documents the choice" from the spec.
- **Review Focus** items each have an owning task and test: leakage (Task 3/4), negative overlap (Task 2), rankable-only candidates (Task 8), cold-start separation (Task 8), reverse edges (Task 3).
