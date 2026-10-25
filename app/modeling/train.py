"""Training loops for the baseline and GNN disease-gene link predictors."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import torch
from torch import nn
from torch_geometric.data import HeteroData

from app.modeling.loader import GraphPackage, NodeIndex
from app.modeling.models import BaselineScorer, GNNLinkPredictor, compute_gene_prior
from app.modeling.scoring import pairs_to_index_tensors


@dataclass
class TrainingHistory:
    """Per-epoch train/validation loss values.

    `best_epoch` is the 1-indexed epoch with the lowest `validation_loss`, tracked
    regardless of `patience`. It only affects which weights the training function
    returns when `patience` is set (see `train_baseline`/`train_gnn`).
    """

    train_loss: list[float] = field(default_factory=list)
    validation_loss: list[float] = field(default_factory=list)
    best_epoch: int = 0


class _BestCheckpoint:
    """Tracks the lowest validation loss seen and a copy of the model's weights then."""

    def __init__(self) -> None:
        self.loss = float("inf")
        self.epoch = 0
        self.state_dict: dict[str, torch.Tensor] | None = None
        self.epochs_since_improvement = 0

    def update(self, model: nn.Module, epoch: int, val_loss: float) -> None:
        if val_loss < self.loss:
            self.loss = val_loss
            self.epoch = epoch
            self.state_dict = copy.deepcopy(model.state_dict())
            self.epochs_since_improvement = 0
        else:
            self.epochs_since_improvement += 1

    def should_stop(self, patience: int | None) -> bool:
        return patience is not None and self.epochs_since_improvement >= patience


def _build_pair_batch(
    package: GraphPackage, split: str
) -> tuple[torch.LongTensor, torch.LongTensor, torch.Tensor]:
    """Build combined (disease_idx, gene_idx, labels) tensors for positives+negatives of a split."""
    disease_index = package.node_indexes["Disease"].index
    gene_index = package.node_indexes["Gene"].index

    positives = package.positives[split]
    negatives = package.negatives[split]
    if not positives and not negatives:
        raise ValueError(
            f"split {split!r} has no positive or negative pairs; refusing to "
            "train/evaluate on an empty batch (BCE loss on an empty batch is NaN, "
            "which would otherwise silently corrupt early stopping)"
        )

    pos_disease_idx, pos_gene_idx = pairs_to_index_tensors(positives, disease_index, gene_index)
    neg_disease_idx, neg_gene_idx = pairs_to_index_tensors(negatives, disease_index, gene_index)

    disease_idx = torch.cat([pos_disease_idx, neg_disease_idx])
    gene_idx = torch.cat([pos_gene_idx, neg_gene_idx])
    labels = torch.cat(
        [torch.ones(len(positives)), torch.zeros(len(negatives))]
    )
    return disease_idx, gene_idx, labels


def train_baseline(
    package: GraphPackage,
    *,
    epochs: int = 50,
    lr: float = 0.01,
    weight_decay: float = 0.0,
    seed: int = 42,
    patience: int | None = None,
) -> tuple[BaselineScorer, TrainingHistory]:
    """Train `BaselineScorer` on `package`'s train split, tracking validation loss too.

    `history.best_epoch` (the 1-indexed epoch with the lowest validation loss) is
    always recorded. If `patience` is given, training stops once that many
    consecutive epochs pass with no new best validation loss, and the returned
    model's weights are rolled back to that best epoch rather than the last one.
    With `patience=None` (the default), training always runs the full `epochs` and
    returns the final epoch's weights, unchanged from prior behavior.

    `weight_decay` adds L2 regularization via Adam's own `weight_decay` argument —
    train positives are far fewer than the model's parameters, so some penalty
    helps curb the fast overfitting seen on this dataset (default 0.0 = off).
    """
    torch.manual_seed(seed)

    disease_ids = package.node_indexes["Disease"].ids
    gene_ids = package.node_indexes["Gene"].ids
    gene_prior = compute_gene_prior(package.positives["train"], gene_ids)

    model = BaselineScorer(disease_ids, gene_ids, gene_prior, seed=seed)

    train_disease_idx, train_gene_idx, train_labels = _build_pair_batch(package, "train")
    val_disease_idx, val_gene_idx, val_labels = _build_pair_batch(package, "validation")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    history = TrainingHistory()
    checkpoint = _BestCheckpoint()

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        train_logits = model(train_disease_idx, train_gene_idx)
        train_loss = loss_fn(train_logits, train_labels)
        train_loss.backward()
        optimizer.step()
        history.train_loss.append(train_loss.item())

        model.eval()
        with torch.no_grad():
            val_logits = model(val_disease_idx, val_gene_idx)
            val_loss = loss_fn(val_logits, val_labels)
        history.validation_loss.append(val_loss.item())

        checkpoint.update(model, epoch + 1, val_loss.item())
        if checkpoint.should_stop(patience):
            break

    history.best_epoch = checkpoint.epoch
    if patience is not None and checkpoint.state_dict is not None:
        model.load_state_dict(checkpoint.state_dict)

    return model, history


DG_EDGE_TYPE: tuple[str, str, str] = ("Disease", "ASSOCIATED_WITH", "Gene")
DG_REVERSE_EDGE_TYPE: tuple[str, str, str] = ("Gene", "rev_ASSOCIATED_WITH", "Disease")
DEFAULT_SUPERVISION_RATIO: float = 0.3


def split_message_supervision(
    train_pairs: list[tuple[str, str]],
    *,
    supervision_ratio: float,
    generator: torch.Generator,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Randomly partition train positives into (message, supervision) subsets.

    ``round(len * supervision_ratio)`` pairs (at least 1 when there is any pair) go to
    the supervision subset, the rest stay as message-passing context. The draw uses
    only ``generator``, never global RNG. Both outputs keep ``train_pairs`` order.
    """
    if not 0.0 < supervision_ratio <= 1.0:
        raise ValueError(f"supervision_ratio must be in (0, 1], got {supervision_ratio}")
    count = len(train_pairs)
    if count == 0:
        return [], []
    num_supervision = min(count, max(1, round(count * supervision_ratio)))
    permutation = torch.randperm(count, generator=generator)
    supervision_positions = set(permutation[:num_supervision].tolist())
    message = [pair for i, pair in enumerate(train_pairs) if i not in supervision_positions]
    supervision = [pair for i, pair in enumerate(train_pairs) if i in supervision_positions]
    return message, supervision


def remove_dg_edges(
    data: HeteroData,
    node_indexes: dict[str, NodeIndex],
    pairs: list[tuple[str, str]],
) -> HeteroData:
    """Return a copy of ``data`` without the given Disease-Gene pairs' edges.

    Removes each pair's forward ``(Disease, ASSOCIATED_WITH, Gene)`` edge AND its
    mirrored ``(Gene, rev_ASSOCIATED_WITH, Disease)`` edge. ``data`` itself is not
    modified. Edge types are kept even if they become empty, so the model's
    metadata still matches.
    """
    result = data.clone()
    if not pairs or DG_EDGE_TYPE not in data.edge_types:
        return result

    disease_index = node_indexes["Disease"].index
    gene_index = node_indexes["Gene"].index
    num_genes = len(node_indexes["Gene"].ids)
    removed_keys = torch.tensor(
        [disease_index[d] * num_genes + gene_index[g] for d, g in pairs], dtype=torch.long
    )

    forward = data[DG_EDGE_TYPE].edge_index
    forward_keys = forward[0] * num_genes + forward[1]
    result[DG_EDGE_TYPE].edge_index = forward[:, ~torch.isin(forward_keys, removed_keys)]

    if DG_REVERSE_EDGE_TYPE in data.edge_types:
        reverse = data[DG_REVERSE_EDGE_TYPE].edge_index
        reverse_keys = reverse[1] * num_genes + reverse[0]
        result[DG_REVERSE_EDGE_TYPE].edge_index = reverse[
            :, ~torch.isin(reverse_keys, removed_keys)
        ]
    return result


@dataclass
class GNNEpochBatch:
    """One GNN training epoch's message graph and BCE batch."""

    message_data: HeteroData
    supervision_pairs: list[tuple[str, str]]
    negative_pairs: list[tuple[str, str]]
    disease_idx: torch.LongTensor
    gene_idx: torch.LongTensor
    labels: torch.Tensor


def build_gnn_epoch_batch(
    package: GraphPackage,
    *,
    supervision_ratio: float,
    generator: torch.Generator,
) -> GNNEpochBatch:
    """Edge-level held-out supervision for one GNN training epoch.

    Train positives are freshly partitioned into message/supervision subsets. The
    supervision pairs become this epoch's BCE positives and their forward + reverse
    ``ASSOCIATED_WITH`` edges are removed from the graph the model encodes, so a
    train positive never reaches its own score through message passing. An equal
    number of train negatives (sampled from ``package.negatives["train"]`` with the
    same generator) keeps the 1:1 class balance of the full train split.
    """
    _message, supervision = split_message_supervision(
        package.positives["train"], supervision_ratio=supervision_ratio, generator=generator
    )
    train_negatives = package.negatives["train"]
    num_negatives = min(len(supervision), len(train_negatives))
    negative_positions = sorted(
        torch.randperm(len(train_negatives), generator=generator)[:num_negatives].tolist()
    )
    negatives = [train_negatives[i] for i in negative_positions]

    disease_index = package.node_indexes["Disease"].index
    gene_index = package.node_indexes["Gene"].index
    pos_disease_idx, pos_gene_idx = pairs_to_index_tensors(supervision, disease_index, gene_index)
    neg_disease_idx, neg_gene_idx = pairs_to_index_tensors(negatives, disease_index, gene_index)

    return GNNEpochBatch(
        message_data=remove_dg_edges(package.data, package.node_indexes, supervision),
        supervision_pairs=supervision,
        negative_pairs=negatives,
        disease_idx=torch.cat([pos_disease_idx, neg_disease_idx]),
        gene_idx=torch.cat([pos_gene_idx, neg_gene_idx]),
        labels=torch.cat([torch.ones(len(supervision)), torch.zeros(len(negatives))]),
    )


def train_gnn(
    package: GraphPackage,
    *,
    epochs: int = 50,
    lr: float = 0.01,
    weight_decay: float = 0.0,
    hidden_dim: int = 64,
    embedding_dim: int = 64,
    seed: int = 42,
    supervision_ratio: float = DEFAULT_SUPERVISION_RATIO,
    patience: int | None = None,
) -> tuple[GNNLinkPredictor, TrainingHistory]:
    """Train `GNNLinkPredictor` on `package`'s train split, tracking validation loss too.

    Uses edge-level held-out supervision (see `build_gnn_epoch_batch`): each epoch,
    a fresh ``supervision_ratio`` fraction of train positives is removed from the
    message graph and used as BCE positives, so the model cannot score a train
    positive by detecting its own edge in the encoded graph. Validation loss (and
    downstream evaluation) encodes the FULL train message graph, the correct
    transductive setup since val/test positives were never in it.

    `history.best_epoch` (the 1-indexed epoch with the lowest validation loss) is
    always recorded. If `patience` is given, training stops once that many
    consecutive epochs pass with no new best validation loss, and the returned
    model's weights are rolled back to that best epoch rather than the last one.
    With `patience=None` (the default), training always runs the full `epochs` and
    returns the final epoch's weights, unchanged from prior behavior.

    `weight_decay` adds L2 regularization via Adam's own `weight_decay` argument.
    `hidden_dim`/`embedding_dim` default to 64. An earlier bilinear decoder
    (`embedding_dim**2` parameters) overfit badly at this width with only 6,673
    train positives and was shrunk to 32 as a workaround; the decoder is now
    DistMult (`embedding_dim` parameters -- see `GNNLinkPredictor`), which
    tolerates 64-dim node embeddings better, confirmed by a grid sweep on the
    real dataset (highest validation MRR at weight_decay=1e-4, dim=64).
    """
    torch.manual_seed(seed)
    split_generator = torch.Generator().manual_seed(seed)

    num_nodes = {
        node_type: len(node_index.ids) for node_type, node_index in package.node_indexes.items()
    }
    model = GNNLinkPredictor(
        package.data.metadata(),
        num_nodes=num_nodes,
        hidden_dim=hidden_dim,
        embedding_dim=embedding_dim,
        seed=seed,
    )

    val_disease_idx, val_gene_idx, val_labels = _build_pair_batch(package, "validation")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    history = TrainingHistory()
    checkpoint = _BestCheckpoint()

    for epoch in range(epochs):
        batch = build_gnn_epoch_batch(
            package, supervision_ratio=supervision_ratio, generator=split_generator
        )
        model.train()
        optimizer.zero_grad()
        embeddings = model.encode(batch.message_data)
        train_logits = model.decode(
            embeddings["Disease"], embeddings["Gene"], batch.disease_idx, batch.gene_idx
        )
        train_loss = loss_fn(train_logits, batch.labels)
        train_loss.backward()
        optimizer.step()
        history.train_loss.append(train_loss.item())

        model.eval()
        with torch.no_grad():
            val_embeddings = model.encode(package.data)
            val_logits = model.decode(
                val_embeddings["Disease"], val_embeddings["Gene"], val_disease_idx, val_gene_idx
            )
            val_loss = loss_fn(val_logits, val_labels)
        history.validation_loss.append(val_loss.item())

        checkpoint.update(model, epoch + 1, val_loss.item())
        if checkpoint.should_stop(patience):
            break

    history.best_epoch = checkpoint.epoch
    if patience is not None and checkpoint.state_dict is not None:
        model.load_state_dict(checkpoint.state_dict)

    return model, history
