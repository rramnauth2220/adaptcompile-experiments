#!/usr/bin/env python3
"""Rebuild the Llama calibration summary reported as Table 1.

The command consumes only the four compact release CSVs in
``artifacts/llama/calibration`` and writes a derived CSV outside the immutable
artifact tree.
"""

from __future__ import annotations

import argparse
import csv
from decimal import Decimal, ROUND_HALF_UP
from fractions import Fraction
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
INPUT_DIR = PROJECT_ROOT / "artifacts" / "llama" / "calibration"
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "outputs"
    / "release_verification"
    / "tables"
    / "llama"
    / "table1_calibration.csv"
)
GRAD_ACCUM_VALUES = (1, 2, 4, 8)
METRICS = ("acquisition", "transfer", "boundedness", "preservation")
PAPER_PRECISION = Decimal("0.001")


def load_calibration_summary(input_dir: Path = INPUT_DIR) -> list[dict[str, object]]:
    """Return one aggregate row per gradient-accumulation schedule."""

    rows = []
    for grad_accum in GRAD_ACCUM_VALUES:
        path = input_dir / f"gradacc_{grad_accum}_geometry.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Missing release calibration source: {path}")
        with path.open(encoding="utf-8", newline="") as handle:
            episodes = list(csv.DictReader(handle))
        if len(episodes) != 25:
            raise ValueError(f"Expected 25 episodes in {path}, found {len(episodes)}")
        required_columns = (*METRICS, "n_preservation_evaluated", "n_preservation_retained")
        missing = [column for column in required_columns if column not in (episodes[0] or {})]
        if missing:
            raise ValueError(f"Missing columns in {path}: {', '.join(missing)}")
        row: dict[str, object] = {
            "grad_accum": grad_accum,
            "number_of_episodes": len(episodes),
        }
        for metric in METRICS:
            if metric == "preservation":
                exact_mean = sum(
                    (
                        Fraction(
                            int(episode["n_preservation_retained"]),
                            int(episode["n_preservation_evaluated"]),
                        )
                        for episode in episodes
                    ),
                    Fraction(),
                ) / len(episodes)
                mean = Decimal(exact_mean.numerator) / Decimal(exact_mean.denominator)
            else:
                mean = sum((Decimal(episode[metric]) for episode in episodes), Decimal()) / len(episodes)
            row[metric] = mean.quantize(PAPER_PRECISION, rounding=ROUND_HALF_UP)
        rows.append(row)
    return rows


def write_table(rows: list[dict[str, object]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("grad_accum", *METRICS, "number_of_episodes"),
        )
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Derived CSV destination (default: outputs/release_verification/tables/llama/).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    rows = load_calibration_summary()
    write_table(rows, output)
    print(f"Wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
