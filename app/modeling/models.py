"""Disease-gene link prediction models.

- `BaselineScorer`: no-graph baseline (gene prior + DistMult disease/gene scorer).
- `GNNLinkPredictor`: 2-layer heterogeneous GNN encoder + DistMult decoder.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv

EdgeType = tuple[str, str, str]


def compute_gene_prior(
    train_positives: list[tuple[str, str]], gene_ids: list[str]
) -> dict[str, float]:
    """Compute normalized train-positive frequency per gene id.

    Each gene's train-positive count is divided by the maximum count across
    all genes, so the most-associated gene gets a prior of 1.0. Genes with no
    train-positive occurrences (including genes unseen in `train_positives`)
    default to 0.0.
    """
    counts: dict[str, int] = {gene_id: 0 for gene_id in gene_ids}
    for _disease_id, gene_id in train_positives:
        if gene_id in counts:
            counts[gene_id] += 1

    max_count = max(counts.values(), default=0)
    if max_count == 0:
        return {gene_id: 0.0 for gene_id in gene_ids}

    return {gene_id: counts[gene_id] / max_count for gene_id in gene_ids}


class BaselineScorer(nn.Module):
    """No-graph baseline: gene prior logit plus a learned DistMult term.

    Combines a fixed, precomputed gene prior (how often a gene appears as a
    positive label in the train split) with a learned DistMult-style scorer
    over small disease/gene embeddings:

        logit = gene_prior_logit[gene] + sum(disease_emb * relation_weight * gene_emb)

    DistMult (a diagonal relation vector) replaces an earlier full bilinear
    matrix (`embedding_dim x embedding_dim`): with only ~6,700 train positives,
    the full matrix (`embedding_dim**2` parameters) overfit badly. The
    diagonal form has `embedding_dim` parameters instead.
    """

    def __init__(
        self,
        disease_ids: list[str],
        gene_ids: list[str],
        gene_prior: dict[str, float],
        *,
        embedding_dim: int = 32,
        seed: int = 42,
    ) -> None:
        super().__init__()

        self.disease_ids = list(disease_ids)
        self.gene_ids = list(gene_ids)
        self.embedding_dim = embedding_dim

        num_diseases = len(self.disease_ids)
        num_genes = len(self.gene_ids)

        # Build weight tensors uninitialized (torch.empty draws no RNG), then
        # fill them using a local generator so construction never touches
        # global torch RNG state. Passing `_weight` to nn.Embedding skips its
        # own (global-RNG-based) reset_parameters call.
        generator = torch.Generator().manual_seed(seed)

        disease_weight = torch.empty(num_diseases, embedding_dim)
        nn.init.normal_(disease_weight, generator=generator)
        self.disease_embedding = nn.Embedding(
            num_diseases, embedding_dim, _weight=disease_weight
        )

        gene_weight = torch.empty(num_genes, embedding_dim)
        nn.init.normal_(gene_weight, generator=generator)
        self.gene_embedding = nn.Embedding(
            num_genes, embedding_dim, _weight=gene_weight
        )

        relation_weight = torch.empty(embedding_dim)
        nn.init.normal_(relation_weight, generator=generator)
        self.relation_weight = nn.Parameter(relation_weight)

        gene_prior_values = torch.tensor(
            [gene_prior.get(gene_id, 0.0) for gene_id in self.gene_ids],
            dtype=torch.float32,
        )
        self.register_buffer("gene_prior", gene_prior_values)

    def forward(
        self, disease_idx: torch.LongTensor, gene_idx: torch.LongTensor
    ) -> torch.Tensor:
        """Return a 1-D logit tensor, one per (disease, gene) input pair."""
        disease_emb = self.disease_embedding(disease_idx)
        gene_emb = self.gene_embedding(gene_idx)

        distmult_term = (disease_emb * self.relation_weight * gene_emb).sum(dim=-1)
        gene_prior_logit = self.gene_prior[gene_idx]

        return gene_prior_logit + distmult_term


class GNNLinkPredictor(nn.Module):
    """Two-layer heterogeneous GNN encoder with a DistMult disease/gene decoder.

    Encoder:
        learned per-node input embedding (per node type, hidden_dim)
        -> HeteroConv(layer 1) -> ReLU -> dropout
        -> HeteroConv(layer 2) -> ReLU
        -> per-type linear projection to embedding_dim (Disease, Gene only)

    Each HeteroConv holds one relation-specific conv per edge type in
    `metadata` -- forward relations AND their `rev_*` mirrors -- and sums the
    per-relation messages arriving at each destination node type.

    Conv choice: `SAGEConv` with mean neighbor aggregation per relation.
    - Its separate root (self) weight preserves a node's own embedding, so
      nodes with no incoming edges of a relation (e.g. a gene with no train
      association) still get a well-defined, non-degenerate representation.
    - Mean aggregation normalizes away the heavy degree skew of hub genes /
      broad phenotypes, unlike GraphConv's sum aggregation.
    - It is cheap and deterministic on CPU, unlike attention (GATConv), which
      adds parameters and variance that a small labeled set cannot support.

    The graph has no node features, so input features are learned per-node
    embeddings (transductive). That requires node counts per type, which
    `HeteroData.metadata()` does not carry -- hence the `num_nodes` argument.

    `decode` has the same shape contract as `BaselineScorer.forward`: one logit
    per (disease_idx[i], gene_idx[i]) pair, as a 1-D tensor.
    """

    DROPOUT: float = 0.1
    OUTPUT_NODE_TYPES: tuple[str, str] = ("Disease", "Gene")

    def __init__(
        self,
        metadata: tuple[list[str], list[EdgeType]],
        *,
        num_nodes: dict[str, int],
        hidden_dim: int = 64,
        embedding_dim: int = 64,
        seed: int = 42,
    ) -> None:
        super().__init__()

        node_types, edge_types = metadata
        self.node_types: list[str] = list(node_types)
        self.edge_types: list[EdgeType] = [tuple(edge_type) for edge_type in edge_types]
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim

        for node_type in self.OUTPUT_NODE_TYPES:
            if node_type not in self.node_types:
                raise ValueError(f"metadata is missing required node type {node_type!r}")
        missing_counts = [t for t in self.node_types if t not in num_nodes]
        if missing_counts:
            raise ValueError(f"num_nodes is missing node types: {missing_counts}")

        # PyG/torch layers initialize themselves from the global RNG on
        # construction. Build them inside fork_rng so global state is restored
        # afterwards, then re-initialize every parameter from a local generator
        # so the model depends only on `seed`.
        with torch.random.fork_rng(devices=[]):
            self.input_embeddings = nn.ModuleDict(
                {
                    node_type: nn.Embedding(num_nodes[node_type], hidden_dim)
                    for node_type in self.node_types
                }
            )
            self.convs = nn.ModuleList(
                [
                    HeteroConv(
                        {
                            edge_type: SAGEConv((hidden_dim, hidden_dim), hidden_dim)
                            for edge_type in self.edge_types
                        },
                        aggr="sum",
                    )
                    for _ in range(2)
                ]
            )
            self.projections = nn.ModuleDict(
                {
                    node_type: nn.Linear(hidden_dim, embedding_dim)
                    for node_type in self.OUTPUT_NODE_TYPES
                }
            )
            # DistMult decoder: a diagonal relation vector (embedding_dim params)
            # rather than a full embedding_dim x embedding_dim bilinear matrix --
            # far fewer parameters to overfit given ~6,700 train positives.
            self.relation_weight = nn.Parameter(torch.empty(embedding_dim))

        self._reset_parameters(torch.Generator().manual_seed(seed))
        # Dropout during training draws Bernoulli masks from this model-owned
        # generator (never the global RNG) so encode() stays reproducible from
        # `seed` alone, matching the parameter-init RNG isolation above.
        self._dropout_generator = torch.Generator().manual_seed(seed)

    def _reset_parameters(self, generator: torch.Generator) -> None:
        """Deterministically (re)initialize all parameters from `generator`."""
        with torch.no_grad():
            for name, param in self.named_parameters():
                if name.startswith("input_embeddings.") or name == "relation_weight":
                    nn.init.normal_(param, generator=generator)
                elif param.dim() >= 2:
                    nn.init.xavier_uniform_(param, generator=generator)
                else:
                    nn.init.zeros_(param)

    def _dropout(self, x: torch.Tensor) -> torch.Tensor:
        """Inverted-dropout using the model-owned generator, never global RNG."""
        if not self.training or self.DROPOUT == 0.0:
            return x
        keep_prob = 1 - self.DROPOUT
        mask = torch.empty_like(x).bernoulli_(keep_prob, generator=self._dropout_generator)
        return x * mask / keep_prob

    def encode(self, data: HeteroData) -> dict[str, torch.Tensor]:
        """Return `{"Disease": [num_disease, D], "Gene": [num_gene, D]}` embeddings."""
        x_dict: dict[str, torch.Tensor] = {}
        for node_type in self.node_types:
            embedding = self.input_embeddings[node_type]
            expected = data[node_type].num_nodes
            if expected != embedding.num_embeddings:
                raise ValueError(
                    f"{node_type}: data has {expected} nodes but model was built for "
                    f"{embedding.num_embeddings}"
                )
            x_dict[node_type] = embedding.weight

        edge_index_dict = data.edge_index_dict
        for layer, conv in enumerate(self.convs):
            out_dict = conv(x_dict, edge_index_dict)
            # A node type receiving no edges this layer is absent from
            # out_dict; carry its previous representation forward.
            x_dict = {
                node_type: F.relu(out_dict.get(node_type, x))
                for node_type, x in x_dict.items()
            }
            if layer < len(self.convs) - 1:
                x_dict = {
                    node_type: self._dropout(x)
                    for node_type, x in x_dict.items()
                }

        return {
            node_type: self.projections[node_type](x_dict[node_type])
            for node_type in self.OUTPUT_NODE_TYPES
        }

    def decode(
        self,
        disease_emb: torch.Tensor,
        gene_emb: torch.Tensor,
        disease_idx: torch.LongTensor,
        gene_idx: torch.LongTensor,
    ) -> torch.Tensor:
        """Return a 1-D logit tensor, one per (disease, gene) input pair."""
        return (
            disease_emb[disease_idx] * self.relation_weight * gene_emb[gene_idx]
        ).sum(dim=-1)
