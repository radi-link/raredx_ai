"""Offline annotation frequencies; intentionally separate from runtime ranking."""

from __future__ import annotations

import math

FREQUENCY_RULE_VERSION = "interval_midpoint_v1"
FREQUENCY_WEIGHTS = {
    "Obligate (100%)": 1.0,
    "Very frequent (99-80%)": 0.895,
    "Frequent (79-30%)": 0.545,
    "Occasional (29-5%)": 0.170,
    "Very rare (<4-1%)": 0.025,
    "Excluded (0%)": 0.0,
}
HPO_FREQUENCIES = dict(
    zip([f"HP:004028{i}" for i in range(6)], FREQUENCY_WEIGHTS.values(), strict=True)
)


def frequency_weight(value: str | None) -> float | None:
    """Preserve unknown as NULL, parse HPO categories, exact percentages and ratios."""
    raw = (value or "").strip()
    if raw in FREQUENCY_WEIGHTS:
        return FREQUENCY_WEIGHTS[raw]
    if raw in HPO_FREQUENCIES:
        return HPO_FREQUENCIES[raw]
    try:
        if "/" in raw:
            numerator, denominator = raw.split("/", 1)
            number = float(numerator) / float(denominator)
        elif raw.endswith("%"):
            number = float(raw[:-1]) / 100
        else:
            return None
    except (ValueError, ZeroDivisionError):
        return None
    return number if math.isfinite(number) and 0 <= number <= 1 else None


def is_negative(qualifier: str | None, frequency: str | None) -> bool:
    return (
        "NOT" in {token.strip() for token in (qualifier or "").upper().split("|")}
        or frequency_weight(frequency) == 0
    )
