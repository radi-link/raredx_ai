"""CSV loading primitives for disease-gene link prediction."""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch_geometric.data import HeteroData

NodeRecord = dict[str, Any]


def _parse_rankable(value: str | None) -> bool:
    """Parse rankable column to bool, defaulting to True if missing or empty."""
    if not value or value.strip() == "":
        return True
    return value.strip().lower() == "true"


def load_node_csv(path: Path) -> dict[str, NodeRecord]:
    """Load node CSV, indexed by node_id, include node_type, rankable, canonical_status."""
    result: dict[str, NodeRecord] = {}
    with path.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row is None:
                continue
            node_id = row.get("node_id")
            if not node_id:
                continue
            # Parse rankable with default True if column missing or empty
            rankable_str = row.get("rankable")
            rankable = _parse_rankable(rankable_str)
            # Create record with parsed rankable value
            record: NodeRecord = {
                **row,
                "rankable": rankable,
            }
            result[node_id] = record
    return result


def load_message_edge_csv(path: Path) -> list[dict[str, Any]]:
    """Load message edge CSV, return all rows as dict."""
    result: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row is None:
                continue
            result.append(row)
    return result


def load_pair_csv(path: Path) -> list[tuple[str, str]]:
    """Load pair CSV, return (source_id, target_id) for rows where relation == 'ASSOCIATED_WITH'.

    Raises ValueError if any other relation is present.
    """
    result: list[tuple[str, str]] = []
    with path.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        relations_seen: set[str] = set()
        for row in reader:
            if row is None:
                continue
            relation = row.get("relation")
            if relation:
                relations_seen.add(relation)
            if relation == "ASSOCIATED_WITH":
                source_id = row.get("source_id")
                target_id = row.get("target_id")
                if source_id and target_id:
                    result.append((source_id, target_id))
        # Check if any non-ASSOCIATED_WITH relations present
        if relations_seen and relations_seen != {"ASSOCIATED_WITH"}:
            raise ValueError(
                f"load_pair_csv expects only ASSOCIATED_WITH relation, found: {relations_seen}"
            )
    return result


def load_negative_pairs_csv(path: Path) -> list[tuple[str, str]]:
    """Load negative pairs CSV, return (source_id, target_id) for rows where label == '0'."""
    result: list[tuple[str, str]] = []
    with path.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row is None:
                continue
            label = row.get("label")
            if label == "0":
                source_id = row.get("source_id")
                target_id = row.get("target_id")
                if source_id and target_id:
                    result.append((source_id, target_id))
    return result


def split_negative_pairs(
    negatives: list[tuple[str, str]],
    *,
    split_sizes: dict[str, int],
    seed: int = 42,
) -> dict[str, list[tuple[str, str]]]:
    """Split negative pairs deterministically into train/validation/test groups.

    Args:
        negatives: List of (source_id, target_id) tuples.
        split_sizes: Dictionary with keys "train", "validation", "test" specifying sizes.
        seed: Random seed for reproducibility (default 42).

    Returns:
        Dictionary with keys "train", "validation", "test" containing the split pairs.

    Raises:
        ValueError: If sum of split sizes exceeds number of negatives.
    """
    total_requested = sum(split_sizes.values())
    if total_requested > len(negatives):
        raise ValueError(
            f"split_sizes sum ({total_requested}) exceeds number of negatives ({len(negatives)})"
        )

    # Sort for stable base order, then shuffle with local RNG
    sorted_pairs = sorted(negatives)
    shuffled_pairs = sorted_pairs[:]
    random.Random(seed).shuffle(shuffled_pairs)

    # Slice sequentially in order: train, validation, test
    train_size = split_sizes.get("train", 0)
    val_size = split_sizes.get("validation", 0)

    result = {
        "train": shuffled_pairs[:train_size],
        "validation": shuffled_pairs[train_size : train_size + val_size],
        "test": shuffled_pairs[train_size + val_size : train_size + val_size + split_sizes.get("test", 0)],
    }

    return result


@dataclass
class NodeIndex:
    """Ordered id list for one node type, plus id->position index and rankable subset."""

    ids: list[str]
    index: dict[str, int]
    rankable: set[str] = field(default_factory=set)


def build_node_index(nodes: dict[str, NodeRecord]) -> NodeIndex:
    """Build a NodeIndex from a dict of node_id -> NodeRecord (as returned by load_node_csv).

    Preserves the insertion order of `nodes` for `ids`, and collects the subset of ids
    whose record has `rankable is True` into `rankable`.
    """
    ids = list(nodes.keys())
    index = {node_id: position for position, node_id in enumerate(ids)}
    rankable = {node_id for node_id, record in nodes.items() if record.get("rankable") is True}
    return NodeIndex(ids=ids, index=index, rankable=rankable)


def _node_type_for_id(node_id: str, node_indexes: dict[str, NodeIndex]) -> str | None:
    """Find which node type's NodeIndex contains this id."""
    for node_type, node_index in node_indexes.items():
        if node_id in node_index.index:
            return node_type
    return None


def build_hetero_data(
    node_indexes: dict[str, NodeIndex], message_edges: list[dict[str, Any]]
) -> HeteroData:
    """Build a PyG HeteroData graph from node indexes and message edges.

    Adds one edge type per relation in `message_edges`, keyed as
    (source_node_type, relation, target_node_type), plus a mirrored reverse edge type
    (target_node_type, f"rev_{relation}", source_node_type) for every relation, so that
    message-passing GNN layers can propagate information in both directions.
    """
    data = HeteroData()

    for node_type, node_index in node_indexes.items():
        data[node_type].num_nodes = len(node_index.ids)

    # Group edges by (relation, source_node_type, target_node_type)
    grouped: dict[tuple[str, str, str], list[tuple[int, int]]] = {}
    for edge in message_edges:
        relation = edge["relation"]
        source_id = edge["source_id"]
        target_id = edge["target_id"]

        source_type = _node_type_for_id(source_id, node_indexes)
        target_type = _node_type_for_id(target_id, node_indexes)
        if source_type is None or target_type is None:
            raise ValueError(
                f"Could not resolve node type for edge {source_id} -{relation}-> {target_id}"
            )

        source_pos = node_indexes[source_type].index[source_id]
        target_pos = node_indexes[target_type].index[target_id]

        key = (relation, source_type, target_type)
        grouped.setdefault(key, []).append((source_pos, target_pos))

    for (relation, source_type, target_type), pairs in grouped.items():
        source_positions = [pair[0] for pair in pairs]
        target_positions = [pair[1] for pair in pairs]
        edge_index = torch.tensor([source_positions, target_positions], dtype=torch.long)
        reverse_edge_index = torch.tensor(
            [target_positions, source_positions], dtype=torch.long
        )

        data[source_type, relation, target_type].edge_index = edge_index
        data[target_type, f"rev_{relation}", source_type].edge_index = reverse_edge_index

    return data


SPLIT_NAMES: tuple[str, str, str] = ("train", "validation", "test")
DG_RELATION = "ASSOCIATED_WITH"


def _format_pairs(pairs: set[tuple[str, str]], limit: int = 5) -> str:
    shown = sorted(pairs)[:limit]
    suffix = f" (+{len(pairs) - limit} more)" if len(pairs) > limit else ""
    return f"{shown}{suffix}"


def _check_pairs_resolve(
    kind: str, groups: dict[str, list[tuple[str, str]]], node_indexes: dict[str, NodeIndex]
) -> None:
    disease_index = node_indexes["Disease"].index
    gene_index = node_indexes["Gene"].index
    for split, pairs in groups.items():
        for disease_id, gene_id in pairs:
            if disease_id not in disease_index:
                raise ValueError(
                    f"{kind} {split} pair ({disease_id!r}, {gene_id!r}): disease id "
                    f"{disease_id!r} is not a Disease node"
                )
            if gene_id not in gene_index:
                raise ValueError(
                    f"{kind} {split} pair ({disease_id!r}, {gene_id!r}): gene id "
                    f"{gene_id!r} is not a Gene node"
                )


def _check_splits_disjoint(kind: str, sets: dict[str, set[tuple[str, str]]]) -> None:
    for i, first in enumerate(SPLIT_NAMES):
        for second in SPLIT_NAMES[i + 1 :]:
            overlap = sets[first] & sets[second]
            if overlap:
                raise ValueError(
                    f"{kind} splits {first!r} and {second!r} overlap on "
                    f"{len(overlap)} pair(s): {_format_pairs(overlap)}"
                )


def validate_positive_invariants(
    node_indexes: dict[str, NodeIndex],
    message_edges: list[dict[str, Any]],
    positives: dict[str, list[tuple[str, str]]],
) -> None:
    """Positive-side anti-leakage invariants; raises ``ValueError`` on the first violation.

    1. Every positive pair's disease id is a ``Disease`` node and gene id a ``Gene``
       node (fail at load time instead of a bare ``KeyError`` deep in training).
    2. The train/validation/test positive splits are pairwise disjoint.
    3. No validation/test positive pair appears as an ``ASSOCIATED_WITH`` edge of the
       message-passing graph (val/test leakage into message passing).
    4. The message-passing ``ASSOCIATED_WITH`` pairs are exactly the train positives.
    """
    _check_pairs_resolve("positive", positives, node_indexes)
    positive_sets = {split: set(positives.get(split, [])) for split in SPLIT_NAMES}
    _check_splits_disjoint("positive", positive_sets)

    message_pairs = {
        (edge["source_id"], edge["target_id"])
        for edge in message_edges
        if edge.get("relation") == DG_RELATION
    }
    for heldout in ("validation", "test"):
        leaked = message_pairs & positive_sets[heldout]
        if leaked:
            raise ValueError(
                f"leakage: {len(leaked)} {heldout} positive pair(s) present as "
                f"{DG_RELATION} edges in the message-passing graph: {_format_pairs(leaked)}"
            )
    if message_pairs != positive_sets["train"]:
        extra = message_pairs - positive_sets["train"]
        missing = positive_sets["train"] - message_pairs
        raise ValueError(
            f"message-passing {DG_RELATION} edges must equal the train positives: "
            f"{len(extra)} edge(s) not in train split {_format_pairs(extra)}, "
            f"{len(missing)} train pair(s) missing from graph {_format_pairs(missing)}"
        )


def validate_negative_invariants(
    node_indexes: dict[str, NodeIndex],
    positives: dict[str, list[tuple[str, str]]],
    negatives: dict[str, list[tuple[str, str]]],
) -> None:
    """Negative-side invariants; raises ``ValueError`` on the first violation.

    Every negative pair resolves to real Disease/Gene nodes; the train/validation/test
    negative splits are pairwise disjoint; and no negative is a known positive of any split.
    """
    _check_pairs_resolve("negative", negatives, node_indexes)
    negative_sets = {split: set(negatives.get(split, [])) for split in SPLIT_NAMES}
    _check_splits_disjoint("negative", negative_sets)

    all_positives: set[tuple[str, str]] = set()
    for pairs in positives.values():
        all_positives.update(pairs)
    for split, pairs in negative_sets.items():
        collision = pairs & all_positives
        if collision:
            raise ValueError(
                f"negative {split!r} split contains {len(collision)} known positive "
                f"pair(s): {_format_pairs(collision)}"
            )


def validate_graph_package_invariants(
    node_indexes: dict[str, NodeIndex],
    message_edges: list[dict[str, Any]],
    positives: dict[str, list[tuple[str, str]]],
    negatives: dict[str, list[tuple[str, str]]],
) -> None:
    """Run both positive- and negative-side anti-leakage checks (see each helper)."""
    validate_positive_invariants(node_indexes, message_edges, positives)
    validate_negative_invariants(node_indexes, positives, negatives)


@dataclass
class GraphPackage:
    """Complete graph package with data, node indexes, and train/validation/test splits."""

    data: HeteroData
    node_indexes: dict[str, NodeIndex]
    positives: dict[str, list[tuple[str, str]]]
    negatives: dict[str, list[tuple[str, str]]]


def load_graph_package(graph_dir: Path, *, seed: int = 42) -> GraphPackage:
    """Load a complete graph package from a CSV directory structure.

    Reads:
    - gene_nodes.csv, disease_nodes.csv, hpo_nodes.csv
    - training_message_edges.csv
    - splits/train.csv, splits/validation.csv, splits/test.csv
    - negative_pairs.csv

    Returns a GraphPackage with HeteroData, node indexes, and positive/negative pairs
    split by train/validation/test.

    Args:
        graph_dir: Directory containing all the CSV files.
        seed: Random seed for splitting negatives (default 42).

    Returns:
        GraphPackage with composed data, node_indexes, positives, and negatives.

    Raises:
        ValueError: if the inputs violate an anti-leakage / referential invariant
            (see `validate_positive_invariants` / `validate_negative_invariants`).
    """
    graph_dir = Path(graph_dir)

    # Load nodes
    genes = build_node_index(load_node_csv(graph_dir / "gene_nodes.csv"))
    diseases = build_node_index(load_node_csv(graph_dir / "disease_nodes.csv"))
    phenotypes = build_node_index(load_node_csv(graph_dir / "hpo_nodes.csv"))
    node_indexes = {"Gene": genes, "Disease": diseases, "Phenotype": phenotypes}

    # Load message edges
    message_edges = load_message_edge_csv(graph_dir / "training_message_edges.csv")

    # Load positive pairs from splits
    positives = {
        "train": load_pair_csv(graph_dir / "splits/train.csv"),
        "validation": load_pair_csv(graph_dir / "splits/validation.csv"),
        "test": load_pair_csv(graph_dir / "splits/test.csv"),
    }

    # Enforce anti-leakage invariants here, not only in the upstream builder.
    # Positive-side checks run before negative splitting so a corrupted positive split
    # surfaces as its own error rather than as a negative-pool sizing error.
    validate_positive_invariants(node_indexes, message_edges, positives)

    # Load and split negative pairs
    all_negatives = load_negative_pairs_csv(graph_dir / "negative_pairs.csv")
    split_sizes = {
        "train": len(positives["train"]),
        "validation": len(positives["validation"]),
        "test": len(positives["test"]),
    }
    negatives = split_negative_pairs(all_negatives, split_sizes=split_sizes, seed=seed)

    validate_negative_invariants(node_indexes, positives, negatives)

    # Build HeteroData
    data = build_hetero_data(node_indexes, message_edges)

    return GraphPackage(
        data=data,
        node_indexes=node_indexes,
        positives=positives,
        negatives=negatives,
    )
