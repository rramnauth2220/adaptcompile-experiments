#!/usr/bin/env python3
"""Generate the canonical release-artifact manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIELDS = (
    "release_path",
    "role",
    "experiment",
    "size_bytes",
    "sha256",
    "source_stage",
    "scorer_version",
    "regeneration_command",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_artifacts(artifacts: Path, output: Path) -> Iterable[Path]:
    root_readme = artifacts / "README.md"
    for path in sorted(artifacts.rglob("*")):
        if path.is_file() and path not in {root_readme, output}:
            yield path


def experiment_for(relative: Path) -> str:
    parts = relative.parts
    if parts[1] != "figures":
        return parts[1]
    if len(parts) >= 4 and parts[3] in {"llama", "gemma", "localization", "synthesis"}:
        return parts[3]
    return "figures"


def role_for(relative: Path) -> str:
    parts = relative.parts
    suffix = relative.suffix.lower()
    if parts[1:3] == ("llama", "calibration"):
        return "calibration_episode_table"
    if parts[1:3] == ("gemma", "calibration"):
        return "calibration_evidence"
    if parts[1] == "figures" and parts[2] in {"main", "appendix"}:
        return "released_figure"
    if parts[1:3] == ("figures", "data"):
        return "figure_source"
    if suffix in {".md", ".tex"}:
        return "analysis_report"
    if suffix in {".json", ".jsonl"}:
        return "metadata_or_records"
    return "numerical_evidence"


def source_stage_for(relative: Path) -> str:
    parts = relative.parts
    if parts[1:3] == ("llama", "calibration"):
        return "authoritative Llama schedule-calibration episodes"
    if parts[1:3] == ("gemma", "calibration"):
        return "independent Gemma schedule calibration"
    if parts[1] == "figures" and parts[2] == "data":
        return "compact paper-figure source"
    if parts[1] == "figures":
        return "current-paper rendered figure"
    if parts[1] == "gemma":
        return "corrected compact Gemma evidence"
    if parts[1] == "llama":
        return "authoritative compact Llama evidence"
    return "compact localization evidence"


def scorer_for(relative: Path) -> str:
    parts = relative.parts
    experiment = experiment_for(relative)
    if parts[1:3] == ("gemma", "calibration"):
        return "current_gemma_calibration"
    if experiment == "gemma":
        return "causal_content_v1"
    if experiment == "llama":
        return "current_llama"
    if experiment == "localization":
        return "objective_specific_release"
    return "not_applicable"


def regeneration_for(relative: Path) -> str:
    parts = relative.parts
    if parts[1:3] == ("llama", "calibration"):
        return "python scripts/reproduce/compiler/make_llama_calibration_table.py"
    if parts[1] == "figures" and "llama" in parts:
        return "python scripts/reproduce/compiler/plot_all_paper_figures.py"
    if parts[1] == "figures" and "gemma" in parts:
        return "python scripts/reproduce/plot_exp6_gemma.py"
    if parts[1] == "figures" and "synthesis" in parts:
        return "python scripts/reproduce/plot_compilation_scope.py"
    if parts[1] == "figures" and "cross_model" in parts:
        return "python scripts/reproduce/make_cross_model_figures.py"
    if parts[1] == "figures" and "localization" in parts:
        return "python scripts/reproduce/make_cross_objective_localization_figure.py"
    if parts[1] == "gemma":
        return "python scripts/experiments/compiler/summarize_gemma_compiler_replication.py"
    if parts[1] == "llama":
        return "python src/prepare_geometry_dataset.py"
    if "cross_model" in parts:
        return "python scripts/reproduce/analyze_cross_model_geometry.py"
    if "parameter_matched" in parts:
        return "python scripts/experiments/localization/run_parameter_matched_localization_controls.py --summary_only"
    return "python scripts/reproduce/make_cross_objective_localization_table.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=PROJECT_ROOT / "artifacts")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "artifacts" / "MANIFEST.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifacts = args.artifacts if args.artifacts.is_absolute() else PROJECT_ROOT / args.artifacts
    output = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in iter_artifacts(artifacts, output):
        relative = path.relative_to(PROJECT_ROOT)
        rows.append(
            {
                "release_path": relative.as_posix(),
                "role": role_for(relative),
                "experiment": experiment_for(relative),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "source_stage": source_stage_for(relative),
                "scorer_version": scorer_for(relative),
                "regeneration_command": regeneration_for(relative),
            }
        )
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} artifact rows to {output}")


if __name__ == "__main__":
    main()
