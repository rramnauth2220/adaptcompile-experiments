#!/usr/bin/env python3
"""Post-hoc additive causal-content rescoring for evaluation.jsonl files."""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from causal_content_scoring import (  # noqa: E402
    CAUSAL_SCORE_VERSION,
    score_causal_content,
    validate_alias_coverage,
)
from common import load_examples_by_id, read_jsonl, write_jsonl  # noqa: E402
from compiler_common import metric_source_for, metric_value, write_csv  # noqa: E402


RESPONSE_KEYS = [
    "response",
    "prediction",
    "generated_text",
    "generation",
    "model_output",
    "completion",
    "output",
]

SIBLING_FILES_TO_MIRROR = [
    "preservation.jsonl",
    "preservation_summary.json",
    "compiler_job_metadata.json",
]


def expand_inputs(patterns: Sequence[str]) -> List[Path]:
    paths: List[Path] = []
    for pattern in patterns:
        matches = glob.glob(pattern, recursive=True)
        if matches:
            paths.extend(Path(match) for match in matches)
        else:
            paths.append(Path(pattern))
    return sorted(set(paths))


def response_text(row: Dict[str, Any]) -> str:
    for key in RESPONSE_KEYS:
        if key in row and row[key] is not None:
            return str(row[key])
    return ""


def is_causal_row(row: Dict[str, Any], example: Dict[str, Any] | None = None) -> bool:
    if row.get("learning_type") == "causal_mapping":
        return True
    if example and example.get("learning_type") == "causal_mapping":
        return True
    scoring = row.get("scoring")
    if isinstance(scoring, dict) and str(scoring.get("scoring_type", "")).startswith("causal_"):
        return True
    return str(row.get("scoring_type", "")).startswith("causal_")


def original_primary_correct(row: Dict[str, Any]) -> float | None:
    split = str(row.get("split", ""))
    try:
        source = metric_source_for("causal_mapping", split)
    except ValueError:
        return None
    return metric_value(row, source)


def corrected_path_for(path: Path, source_root: Path, output_root: Path) -> Path:
    try:
        rel = path.resolve().relative_to(source_root.resolve())
    except ValueError:
        rel = path.name
    return output_root / rel


def mirror_sibling_files(source_evaluation: Path, corrected_evaluation: Path, dry_run: bool) -> int:
    copied = 0
    for name in SIBLING_FILES_TO_MIRROR:
        source = source_evaluation.with_name(name)
        if not source.exists():
            continue
        copied += 1
        if dry_run:
            continue
        destination = corrected_evaluation.with_name(name)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
    return copied


def rescore_rows(
    rows: Sequence[Dict[str, Any]],
    examples_by_id: Dict[str, Dict[str, Any]],
    score_version: str,
) -> tuple[List[Dict[str, Any]], Dict[str, int]]:
    out: List[Dict[str, Any]] = []
    stats = {
        "n_rows": 0,
        "n_causal_rows": 0,
        "n_non_causal_rows": 0,
        "n_changed_relative_to_original_primary_score": 0,
    }
    for row in rows:
        stats["n_rows"] += 1
        example = examples_by_id.get(str(row.get("example_id", "")))
        if not is_causal_row(row, example):
            stats["n_non_causal_rows"] += 1
            out.append(dict(row))
            continue
        if example is None:
            raise ValueError(f"Missing source example for causal row example_id={row.get('example_id')!r}")
        stats["n_causal_rows"] += 1
        score = score_causal_content(example, response_text(row), score_version=score_version)
        repaired = dict(row)
        repaired.update(score.as_fields())
        original = original_primary_correct(row)
        if original is not None and int(float(original)) != int(bool(score.content_correct)):
            stats["n_changed_relative_to_original_primary_score"] += 1
        out.append(repaired)
    return out, stats


def write_manifest(path: Path, rows: List[Dict[str, Any]]) -> None:
    write_csv(
        path,
        rows,
        [
            "original_path",
            "corrected_path",
            "n_rows",
            "n_causal_rows",
            "n_non_causal_rows",
            "n_changed_relative_to_original_primary_score",
            "n_sibling_files_mirrored",
            "score_version",
            "action",
        ],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Add corrected causal-content scores to evaluation JSONLs.")
    parser.add_argument("--input_glob", nargs="+", required=True, help="evaluation.jsonl file(s) or globs.")
    parser.add_argument("--examples_path", default="data/prompt_examples.jsonl")
    parser.add_argument("--source_root", default=None, help="Root used to preserve relative paths.")
    parser.add_argument("--output_root", required=True, help="Parallel tree for corrected outputs.")
    parser.add_argument("--manifest_csv", default=None)
    parser.add_argument("--score_version", default=CAUSAL_SCORE_VERSION, choices=[CAUSAL_SCORE_VERSION])
    parser.add_argument("--max_files", type=int, default=None)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument(
        "--skip_non_causal",
        action="store_true",
        help="Do not copy non-causal evaluation files into the corrected tree.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = [path for path in expand_inputs(args.input_glob) if path.exists()]
    if args.max_files is not None:
        paths = paths[: args.max_files]
    if not paths:
        raise ValueError("No input files found.")

    examples_by_id = load_examples_by_id(args.examples_path)
    causal_examples = [row for row in examples_by_id.values() if row.get("learning_type") == "causal_mapping"]
    validate_alias_coverage(causal_examples)

    if args.source_root:
        source_root = Path(args.source_root)
    else:
        source_root = Path(os.path.commonpath([str(path.resolve().parent) for path in paths]))

    output_root = Path(args.output_root)
    manifest_path = Path(args.manifest_csv) if args.manifest_csv else output_root / "causal_rescore_manifest.csv"
    manifest_rows: List[Dict[str, Any]] = []

    for path in paths:
        rows = read_jsonl(path)
        has_causal = any(
            is_causal_row(row, examples_by_id.get(str(row.get("example_id", ""))))
            for row in rows
        )
        corrected_path = corrected_path_for(path, source_root, output_root)
        action = "rescore_causal" if has_causal else "copy_non_causal"
        if not has_causal and args.skip_non_causal:
            action = "skip_non_causal"
            stats = {
                "n_rows": len(rows),
                "n_causal_rows": 0,
                "n_non_causal_rows": len(rows),
                "n_changed_relative_to_original_primary_score": 0,
            }
            n_sibling_files_mirrored = 0
        elif has_causal:
            corrected_rows, stats = rescore_rows(rows, examples_by_id, args.score_version)
            if not args.dry_run:
                write_jsonl(corrected_path, corrected_rows)
            n_sibling_files_mirrored = mirror_sibling_files(path, corrected_path, args.dry_run)
        else:
            stats = {
                "n_rows": len(rows),
                "n_causal_rows": 0,
                "n_non_causal_rows": len(rows),
                "n_changed_relative_to_original_primary_score": 0,
            }
            if not args.dry_run:
                corrected_path.parent.mkdir(parents=True, exist_ok=True)
                corrected_path.write_bytes(path.read_bytes())
            n_sibling_files_mirrored = mirror_sibling_files(path, corrected_path, args.dry_run)

        manifest_rows.append(
            {
                "original_path": str(path),
                "corrected_path": str(corrected_path),
                **stats,
                "n_sibling_files_mirrored": n_sibling_files_mirrored,
                "score_version": args.score_version,
                "action": action,
            }
        )

    if not args.dry_run:
        write_manifest(manifest_path, manifest_rows)
        print(f"Wrote {manifest_path}")
    print(
        json.dumps(
            {
                "n_files": len(manifest_rows),
                "n_causal_files": sum(1 for row in manifest_rows if row["action"] == "rescore_causal"),
                "n_causal_rows": sum(int(row["n_causal_rows"]) for row in manifest_rows),
                "n_changed_relative_to_original_primary_score": sum(
                    int(row["n_changed_relative_to_original_primary_score"]) for row in manifest_rows
                ),
                "dry_run": bool(args.dry_run),
                "manifest_csv": str(manifest_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
