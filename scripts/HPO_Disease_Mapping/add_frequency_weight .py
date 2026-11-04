#!/usr/bin/env python3
"""Add a numeric frequency_weight column to the Orphanet disease-HPO baseline CSV."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

DEFAULT_INPUT = Path("data/processed/HPO_disease_mapping_baseline.csv")
DEFAULT_OUTPUT = Path("data/processed/HPO_disease_mapping_with_frequency_weight.csv")
OUTPUT_COLUMN = "frequency_weight"

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from app.etl.frequency import frequency_weight  # noqa: E402


def add_frequency_weight(input_csv: Path, output_csv: Path, decimals: int) -> int:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    row_count = 0

    with input_csv.open("r", newline="", encoding="utf-8") as input_file:
        reader = csv.DictReader(input_file)
        if reader.fieldnames is None:
            raise ValueError(f"Input CSV has no header: {input_csv}")

        fieldnames = list(reader.fieldnames)
        if OUTPUT_COLUMN not in fieldnames:
            fieldnames.append(OUTPUT_COLUMN)

        with output_csv.open("w", newline="", encoding="utf-8") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=fieldnames)
            writer.writeheader()

            for row in reader:
                frequency_label = row.get("frequency_label", "").strip()
                weight = frequency_weight(frequency_label)
                row[OUTPUT_COLUMN] = "" if weight is None else f"{weight:.{decimals}f}"
                writer.writerow(row)
                row_count += 1

    return row_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add numeric frequency_weight to Orphanet disease-HPO CSV."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--decimals", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"Input CSV not found: {args.input}")

    row_count = add_frequency_weight(args.input, args.output, args.decimals)
    print(f"Wrote {row_count:,} rows to {args.output}")


if __name__ == "__main__":
    main()
