"""Canonical identifier resolution without symbol/alias guessing."""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from pathlib import Path

from app.etl.hpo_loader import load_hpo_obo
from app.etl.models import PhenotypeTerm


def canonical_disease_id(value: str | None) -> str | None:
    match = re.fullmatch(r"(?:ORPHA:)?([0-9]+)", (value or "").strip())
    return f"ORPHA:{int(match[1])}" if match and int(match[1]) > 0 else None


def canonical_gene(hgnc: str | None, source_key: str) -> dict[str, str | bool]:
    if hgnc:
        match = re.fullmatch(r"(?:HGNC:)?([0-9]+)(?:[.]0+)?", hgnc.strip())
        if not match or int(match[1]) <= 0:
            raise ValueError(f"Invalid HGNC identifier: {hgnc!r}")
        return {
            "gene_id": f"HGNC:{int(match[1])}",
            "canonical_status": "resolved",
            "rankable": True,
        }
    if not source_key or not re.fullmatch(r"[A-Za-z0-9_.-]+", source_key):
        raise ValueError(
            "An unresolved Orphanet gene requires a non-empty source locus key"
        )
    return {
        "gene_id": f"ORPHANET_LOCUS:{source_key}",
        "canonical_status": "unresolved",
        "rankable": False,
    }


@dataclass(frozen=True)
class HPOResolution:
    original_id: str
    canonical_id: str | None
    method: str
    reason: str | None = None


@dataclass
class HPOCatalog:
    terms: dict[str, PhenotypeTerm]
    metadata: dict[str, dict[str, list[str]]]
    alt_to_active: dict[str, str]
    release: str | None

    def resolve(self, value: str | None) -> HPOResolution:
        raw = (value or "").strip()
        current = raw
        seen: set[str] = set()
        method = "active"
        while current not in seen:
            seen.add(current)
            if current in self.terms:
                return HPOResolution(raw, current, method)
            if current in self.alt_to_active:
                return HPOResolution(
                    raw,
                    self.alt_to_active[current],
                    "alt_id" if method == "active" else method,
                )
            row = self.metadata.get(current)
            if row is None:
                return HPOResolution(raw, None, "unresolved", "unknown_id")
            replacements = row.get("replaced_by", [])
            if len(replacements) == 1:
                method = "replaced_by"
                current = replacements[0].split()[0]
                continue
            reason = (
                "multiple_replacements"
                if replacements
                else "consider_only"
                if row.get("consider")
                else "obsolete_without_replacement"
            )
            return HPOResolution(raw, None, "unresolved", reason)
        return HPOResolution(raw, None, "unresolved", "replacement_cycle")


def load_hpo_catalog(path: Path) -> HPOCatalog:
    terms = {
        identifier: replace(
            term, parents=tuple(parent.split()[0] for parent in term.parents)
        )
        for identifier, term in load_hpo_obo(path).items()
    }
    metadata: dict[str, dict[str, list[str]]] = {}
    current: dict[str, list[str]] | None = None
    release = None

    def flush() -> None:
        if current and current.get("id"):
            metadata[current["id"][0]] = current

    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line.startswith("data-version: "):
                release = line.removeprefix("data-version: ")
            if line.startswith("["):
                flush()
                current = {} if line == "[Term]" else None
            elif current is not None and ": " in line:
                key, value = line.split(": ", 1)
                current.setdefault(key, []).append(value)
    flush()
    alt: dict[str, str] = {}
    for hpo_id in terms:
        for alias in metadata[hpo_id].get("alt_id", []):
            if alias in alt and alt[alias] != hpo_id:
                raise ValueError(f"Conflicting active HPO alias: {alias}")
            alt[alias] = hpo_id
    return HPOCatalog(terms, metadata, alt, release)
