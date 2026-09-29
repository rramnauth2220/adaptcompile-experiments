#!/usr/bin/env python3
"""Regenerate only the Experiment 1 appendix figures from saved results."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List


def run_command(cmd: List[str]) -> None:
    print("Running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot only Experiment 1 appendix figures A1-A3.")
    parser.add_argument("--geometry", default="artifacts/llama/geometry/geometry_dataset.csv")
    parser.add_argument(
        "--margins_csv",
        default="artifacts/llama/geometry/headroom/headroom_top2_margins.csv",
    )
    parser.add_argument("--raw_seed_records", default="artifacts/llama/geometry/robustness/geometry_records_3seed.csv")
    parser.add_argument("--robustness_analysis_root", default="artifacts/llama/geometry/robustness")
    parser.add_argument("--output_root", default="outputs/release_verification/figures/llama")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    output_root = Path(args.output_root)
    (output_root / "appendix").mkdir(parents=True, exist_ok=True)
    (output_root / "data").mkdir(parents=True, exist_ok=True)

    commands = [
        [
            sys.executable,
            str(script_dir / "plot_appendix_exp1_margins.py"),
            "--geometry",
            args.geometry,
            "--margins_csv",
            args.margins_csv,
            "--output_root",
            args.output_root,
        ],
        [
            sys.executable,
            str(script_dir / "plot_appendix_exp1_seed_robustness.py"),
            "--raw_records",
            args.raw_seed_records,
            "--analysis_root",
            args.robustness_analysis_root,
            "--output_root",
            args.output_root,
        ],
        [
            sys.executable,
            str(script_dir / "plot_appendix_exp1_outcome_sensitivity.py"),
            "--geometry",
            args.geometry,
            "--output_root",
            args.output_root,
        ],
    ]
    for cmd in commands:
        run_command(cmd)


if __name__ == "__main__":
    main()
