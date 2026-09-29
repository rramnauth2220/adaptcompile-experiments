#!/usr/bin/env python3
"""Build clean episode x configuration x seed geometry records."""

from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

try:  # pragma: no cover
    from .compiler_common import (
        BOUND_SOURCE,
        balanced_utility,
        metric_source_for,
        metric_value,
        read_jsonl,
        write_csv,
        write_jsonl,
    )
    from .causal_content_scoring import CAUSAL_SCORE_VERSION
except ImportError:  # pragma: no cover
    from compiler_common import (
        BOUND_SOURCE,
        balanced_utility,
        metric_source_for,
        metric_value,
        read_jsonl,
        write_csv,
        write_jsonl,
    )
    from causal_content_scoring import CAUSAL_SCORE_VERSION


def expand_inputs(inputs: Sequence[str]) -> List[Path]:
    paths: List[Path] = []
    for value in inputs:
        matches = glob.glob(value, recursive=True)
        if matches:
            paths.extend(Path(m) for m in matches)
        else:
            paths.append(Path(value))
    return sorted(set(paths))


def mean(values: Iterable[float]) -> float | None:
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def load_metadata_for_result(path: Path) -> Dict[str, Any]:
    candidates = [
        path.with_name("compiler_job_metadata.json"),
        path.parent / "compiler_job_metadata.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return json.loads(candidate.read_text(encoding="utf-8"))
    return {}


def metadata_value(
    row: Dict[str, Any],
    metadata: Dict[str, Any],
    key: str,
    default: Any = None,
) -> Any:
    if row.get(key) not in (None, ""):
        return row.get(key)
    return metadata.get(key, default)


def corrected_metric_source_for(
    learning_type: str,
    split: str,
    causal_metric_version: str | None,
) -> str:
    if learning_type == "causal_mapping" and causal_metric_version:
        return "causal_content_correct"
    return metric_source_for(learning_type, split)


def aggregate_file(path: Path, causal_metric_version: str | None = None) -> List[Dict[str, Any]]:
    metadata = load_metadata_for_result(path)
    buckets: Dict[Tuple[str, str, int], Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    meta_by_key: Dict[Tuple[str, str, int], Dict[str, Any]] = {}

    for row in read_jsonl(path):
        learning_type = metadata_value(row, metadata, "learning_type", row.get("learning_type"))
        if learning_type not in BOUND_SOURCE:
            continue
        seed_value = metadata_value(row, metadata, "seed", row.get("compiler_seed", row.get("localization_seed", 0)))
        seed = int(seed_value)
        episode_id = metadata_value(row, metadata, "episode_id")
        config_id = metadata_value(row, metadata, "config_id")
        if not episode_id or not config_id:
            # Raw AAAI rows are not clean compiler records.
            continue
        split = str(row.get("split", ""))
        try:
            source = corrected_metric_source_for(str(learning_type), split, causal_metric_version)
        except ValueError:
            continue
        if (
            learning_type == "causal_mapping"
            and causal_metric_version
            and row.get("causal_score_version") != causal_metric_version
        ):
            raise ValueError(
                f"{path} contains causal row without requested causal_score_version={causal_metric_version!r}. "
                "Run scripts/audits/rescore_causal_evaluations.py first."
            )
        value = metric_value(row, source)
        if value is None:
            continue
        key = (str(episode_id), str(config_id), seed)
        buckets[key][split].append(float(value))
        meta_by_key.setdefault(
            key,
            {
                "episode_id": episode_id,
                "spec_id": row.get("spec_id") or (metadata.get("spec_ids") or [None])[0],
                "meta_split": metadata_value(row, metadata, "meta_split"),
                "learning_type": learning_type,
                "model_name": metadata_value(row, metadata, "model_name"),
                "model_slug": metadata_value(row, metadata, "model_slug"),
                "seed": seed,
                "config_id": config_id,
                "trainable_parameters": metadata.get("trainable_parameters"),
                "parameter_cost": metadata.get("approximate_parameter_cost"),
                "source_file": str(path),
            },
        )

    records: List[Dict[str, Any]] = []
    for key, by_split in sorted(buckets.items()):
        meta = meta_by_key[key]
        id_eval = mean(by_split.get("id_eval", []))
        paraphrase = mean(by_split.get("paraphrase_eval", []))
        acquisition = (id_eval + paraphrase) / 2.0 if id_eval is not None and paraphrase is not None else None
        transfer = mean(by_split.get("generalization", []))
        boundedness = mean(by_split.get("negative_control", []))
        record = {
            **meta,
            "id_eval": id_eval,
            "paraphrase_eval": paraphrase,
            "acquisition": acquisition,
            "transfer": transfer,
            "generalization": transfer,
            "boundedness": boundedness,
            "preservation": None,
            "exploratory_balanced_utility": None,
            "boundedness_metric_source": BOUND_SOURCE.get(str(meta["learning_type"])),
            "causal_metric_source": (
                "causal_content_correct"
                if str(meta["learning_type"]) == "causal_mapping" and causal_metric_version
                else None
            ),
            "causal_metric_version": (
                causal_metric_version
                if str(meta["learning_type"]) == "causal_mapping" and causal_metric_version
                else None
            ),
            "n_id_eval": len(by_split.get("id_eval", [])),
            "n_paraphrase_eval": len(by_split.get("paraphrase_eval", [])),
            "n_generalization": len(by_split.get("generalization", [])),
            "n_negative_control": len(by_split.get("negative_control", [])),
            "n_preservation_total": None,
            "n_preservation_baseline_correct": None,
            "n_preservation_evaluated": None,
            "n_preservation_retained": None,
            "preservation_source_file": None,
        }
        record["exploratory_balanced_utility"] = balanced_utility(record)
        records.append(record)
    return records


def read_preservation_csv(path: str | Path | None) -> Dict[Tuple[str, str, int], Dict[str, Any]]:
    if not path:
        return {}
    out: Dict[Tuple[str, str, int], Dict[str, Any]] = {}
    import csv

    with open(path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            key = (row["episode_id"], row["config_id"], int(row["seed"]))
            out[key] = {
                "preservation": float(row["preservation"]),
                "n_preservation_total": int(float(row.get("n_preservation_total") or 0)) or None,
                "n_preservation_baseline_correct": int(float(row.get("n_preservation_baseline_correct") or 0)) or None,
                "n_preservation_evaluated": int(float(row.get("n_preservation_evaluated") or 0)) or None,
                "n_preservation_retained": int(float(row.get("n_preservation_retained") or 0)) or None,
                "preservation_source_file": str(path),
            }
    return out


def read_preservation_summary_json(path: str | Path) -> Dict[Tuple[str, str, int], Dict[str, Any]]:
    path = Path(path)
    row = json.loads(path.read_text(encoding="utf-8"))
    key = (str(row["episode_id"]), str(row["config_id"]), int(row["seed"]))
    n_baseline = int(row.get("n_preservation_baseline_correct") or 0)
    n_evaluated = int(row.get("n_preservation_evaluated") or 0)
    if n_baseline != n_evaluated:
        raise ValueError(
            f"{path} has n_preservation_evaluated={n_evaluated}, "
            f"expected n_preservation_baseline_correct={n_baseline}."
        )
    return {
        key: {
            "episode_id": key[0],
            "config_id": key[1],
            "seed": key[2],
            "model_name": row.get("model_name"),
            "preservation": float(row["preservation"]),
            "n_preservation_total": int(row.get("n_preservation_total") or 0) or None,
            "n_preservation_baseline_correct": n_baseline or None,
            "n_preservation_evaluated": n_evaluated or None,
            "n_preservation_retained": int(row.get("n_preservation_retained") or 0),
            "preservation_source_file": str(path),
        }
    }


def read_preservation_jsonl(path: str | Path) -> Dict[Tuple[str, str, int], Dict[str, Any]]:
    buckets: Dict[Tuple[str, str, int], List[Dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(path):
        episode_id = row.get("episode_id")
        config_id = row.get("config_id")
        seed = row.get("seed")
        if episode_id in (None, "") or config_id in (None, "") or seed in (None, ""):
            continue
        buckets[(str(episode_id), str(config_id), int(seed))].append(row)

    out: Dict[Tuple[str, str, int], Dict[str, Any]] = {}
    for key, rows in buckets.items():
        correct = [
            float(row["preservation_correct"])
            for row in rows
            if row.get("preservation_correct") not in (None, "")
        ]
        preservation_values = [
            float(row["preservation"])
            for row in rows
            if row.get("preservation") not in (None, "")
        ]
        preservation = mean(correct) if correct else mean(preservation_values)
        if preservation is None:
            continue
        first = rows[0]
        out[key] = {
            "preservation": preservation,
            "n_preservation_total": int(first.get("n_preservation_total") or len(rows)),
            "n_preservation_baseline_correct": int(
                first.get("n_preservation_baseline_correct") or len(rows)
            ),
            "n_preservation_evaluated": int(first.get("n_preservation_evaluated") or len(rows)),
            "n_preservation_retained": int(
                first.get("n_preservation_retained")
                or sum(float(row.get("preservation_correct") or 0) for row in rows)
            ),
            "preservation_source_file": str(path),
        }
    return out


def merge_preservation_stats(
    sources: Sequence[Dict[Tuple[str, str, int], Dict[str, Any]]],
) -> Dict[Tuple[str, str, int], Dict[str, Any]]:
    merged: Dict[Tuple[str, str, int], Dict[str, Any]] = {}
    for source in sources:
        merged.update(source)
    return merged


def derived_preservation_paths(evaluation_inputs: Sequence[Path]) -> List[Path]:
    paths = []
    for path in evaluation_inputs:
        candidate = path.with_name("preservation.jsonl")
        summary_candidate = path.with_name("preservation_summary.json")
        if candidate.exists() and not summary_candidate.exists():
            paths.append(candidate)
    return sorted(set(paths))


def derived_preservation_summary_paths(evaluation_inputs: Sequence[Path]) -> List[Path]:
    paths = []
    for path in evaluation_inputs:
        candidate = path.with_name("preservation_summary.json")
        if candidate.exists():
            paths.append(candidate)
    return sorted(set(paths))


def build_records(
    inputs: Sequence[str],
    preservation_summary: str | Path | None = None,
    preservation_inputs: Sequence[str] | None = None,
    causal_metric_version: str | None = None,
) -> List[Dict[str, Any]]:
    if causal_metric_version not in (None, "", CAUSAL_SCORE_VERSION):
        raise ValueError(f"Unsupported causal_metric_version={causal_metric_version!r}")
    records: List[Dict[str, Any]] = []
    input_paths = expand_inputs(inputs)
    for path in input_paths:
        if path.exists():
            records.extend(aggregate_file(path, causal_metric_version=causal_metric_version or None))
        else:
            print(f"[warn] Missing input: {path}")

    preservation_sources = [read_preservation_csv(preservation_summary)]
    if preservation_inputs:
        preservation_paths = [path for path in expand_inputs(preservation_inputs) if path.exists()]
    else:
        preservation_paths = derived_preservation_paths(input_paths)
    preservation_sources.extend(read_preservation_jsonl(path) for path in preservation_paths)
    preservation_summary_paths = derived_preservation_summary_paths(input_paths)
    preservation_sources.extend(read_preservation_summary_json(path) for path in preservation_summary_paths)
    preservation = merge_preservation_stats(preservation_sources)
    for record in records:
        key = (record["episode_id"], record["config_id"], int(record["seed"]))
        if key in preservation:
            stats = preservation[key]
            if str(stats.get("episode_id", record["episode_id"])) != str(record["episode_id"]):
                raise ValueError(f"Preservation episode_id mismatch for {key}.")
            if str(stats.get("config_id", record["config_id"])) != str(record["config_id"]):
                raise ValueError(f"Preservation config_id mismatch for {key}.")
            if int(stats.get("seed", record["seed"])) != int(record["seed"]):
                raise ValueError(f"Preservation seed mismatch for {key}.")
            if stats.get("model_name") not in (None, "", record.get("model_name")):
                raise ValueError(
                    f"Preservation model_name mismatch for {key}: "
                    f"{stats.get('model_name')!r} != {record.get('model_name')!r}"
                )
            record["preservation"] = stats["preservation"]
            record["n_preservation_total"] = stats.get("n_preservation_total")
            record["n_preservation_baseline_correct"] = stats.get("n_preservation_baseline_correct")
            record["n_preservation_evaluated"] = stats.get("n_preservation_evaluated")
            record["n_preservation_retained"] = stats.get("n_preservation_retained")
            record["preservation_source_file"] = stats.get("preservation_source_file")
            record["exploratory_balanced_utility"] = balanced_utility(record)

    return sorted(records, key=lambda r: (r["episode_id"], r["config_id"], int(r["seed"])))


FIELDS = [
    "episode_id",
    "spec_id",
    "meta_split",
    "learning_type",
    "model_name",
    "model_slug",
    "seed",
    "config_id",
    "id_eval",
    "paraphrase_eval",
    "acquisition",
    "transfer",
    "generalization",
    "boundedness",
    "preservation",
    "exploratory_balanced_utility",
    "trainable_parameters",
    "parameter_cost",
    "boundedness_metric_source",
    "causal_metric_source",
    "causal_metric_version",
    "n_id_eval",
    "n_paraphrase_eval",
    "n_generalization",
    "n_negative_control",
    "n_preservation_total",
    "n_preservation_baseline_correct",
    "n_preservation_evaluated",
    "n_preservation_retained",
    "preservation_source_file",
    "source_file",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build compiler geometry records.")
    parser.add_argument("--inputs", nargs="+", required=True, help="Compiler evaluation JSONLs or globs.")
    parser.add_argument("--output_jsonl", default="outputs/compiler/geometry_records.jsonl")
    parser.add_argument("--output_csv", default="outputs/compiler/geometry_records.csv")
    parser.add_argument("--preservation_summary", default=None)
    parser.add_argument("--preservation_inputs", nargs="+", default=None)
    parser.add_argument(
        "--causal_metric_version",
        default=None,
        choices=[CAUSAL_SCORE_VERSION],
        help="Use corrected causal content metric for causal_mapping rows only.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = build_records(
        args.inputs,
        preservation_summary=args.preservation_summary,
        preservation_inputs=args.preservation_inputs,
        causal_metric_version=args.causal_metric_version,
    )
    write_jsonl(args.output_jsonl, records)
    write_csv(args.output_csv, records, FIELDS)
    print(f"Wrote {len(records)} geometry records to {args.output_jsonl} and {args.output_csv}")


if __name__ == "__main__":
    main()
