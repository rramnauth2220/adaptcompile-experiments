#!/usr/bin/env python3
"""Prepare seed-aggregated geometry labels for Experiment 2."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

try:  # pragma: no cover
    from .compiler_common import read_jsonl, write_csv
except ImportError:  # pragma: no cover
    from compiler_common import read_jsonl, write_csv


PRIMARY_CONFIGS = [
    "early__all__r16",
    "middle__all__r16",
    "late__all__r16",
    "full__all__r4",
]

OUTCOME_COLUMNS = ["acquisition", "transfer", "boundedness", "preservation"]


def read_table(path: str | Path) -> List[Dict[str, Any]]:
    path = Path(path)
    if path.suffix == ".jsonl":
        return read_jsonl(path)
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def as_float(value: Any, *, field: str) -> float:
    if value in (None, ""):
        raise ValueError(f"Missing required geometry value for {field}.")
    return float(value)


def mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values]
    if not vals:
        raise ValueError("Cannot average an empty value list.")
    return sum(vals) / len(vals)


def stable_unique(values: Sequence[Any], *, field: str, key: Tuple[str, ...]) -> Any:
    cleaned = [v for v in values if v not in (None, "")]
    unique = sorted({str(v) for v in cleaned})
    if len(unique) > 1:
        raise ValueError(f"Inconsistent {field} for {key}: {unique}")
    return unique[0] if unique else ""


def validate_episode_split_disjoint(rows: Sequence[Dict[str, Any]]) -> None:
    splits_by_episode: Dict[Tuple[str, str], set[str]] = defaultdict(set)
    for row in rows:
        splits_by_episode[(str(row.get("model_slug") or "model"), str(row["episode_id"]))].add(str(row["meta_split"]))
    overlapping = {
        episode_id: sorted(splits)
        for episode_id, splits in splits_by_episode.items()
        if len(splits) > 1
    }
    if overlapping:
        first = next(iter(overlapping.items()))
        raise ValueError(f"Episode appears in multiple meta_splits: {first}")


def parse_seed_values(values: Sequence[int] | None) -> set[int] | None:
    if values is None:
        return None
    return {int(value) for value in values}


def expected_seed_map(
    train: Sequence[int] | None,
    validation: Sequence[int] | None,
    test: Sequence[int] | None,
) -> Dict[str, set[int] | None]:
    return {
        "train": parse_seed_values(train),
        "validation": parse_seed_values(validation),
        "test": parse_seed_values(test),
    }


def validate_no_duplicate_seed_records(rows: Sequence[Dict[str, Any]]) -> None:
    seen: set[Tuple[str, str, str, int]] = set()
    for row in rows:
        key = (
            str(row.get("model_slug") or "model"),
            str(row["episode_id"]),
            str(row["config_id"]),
            int(float(row["seed"])),
        )
        if key in seen:
            raise ValueError(f"Duplicate raw geometry row for model/episode/config/seed: {key}")
        seen.add(key)


def validate_strict_coverage(
    records: Sequence[Dict[str, Any]],
    config_ids: Sequence[str],
    expected_seeds_by_split: Dict[str, set[int] | None],
) -> None:
    expected_configs = set(config_ids)
    grouped: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[
            (
                str(row.get("model_slug") or "model"),
                str(row["meta_split"]),
                str(row["episode_id"]),
            )
        ].append(row)

    for key, rows in sorted(grouped.items()):
        present_configs = {str(row["config_id"]) for row in rows}
        if present_configs != expected_configs:
            raise ValueError(
                f"Episode lacks complete config coverage for {key}: "
                f"missing={sorted(expected_configs - present_configs)}, "
                f"extra={sorted(present_configs - expected_configs)}"
            )
        split = key[1]
        expected_seeds = expected_seeds_by_split.get(split)
        if expected_seeds is not None:
            for row in rows:
                seeds = {int(value) for value in str(row["seed_values"]).split("|") if value}
                if seeds != expected_seeds:
                    raise ValueError(
                        f"Unexpected seed coverage for {key}/{row['config_id']}: "
                        f"observed={sorted(seeds)}, expected={sorted(expected_seeds)}"
                    )


def prepare_geometry_dataset(
    rows: Sequence[Dict[str, Any]],
    config_ids: Sequence[str] = PRIMARY_CONFIGS,
    expected_seeds_by_split: Dict[str, set[int] | None] | None = None,
    strict: bool = True,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    required = {"episode_id", "config_id", "meta_split", "learning_type", "seed", *OUTCOME_COLUMNS}
    missing = required - set(rows[0]) if rows else set()
    if missing:
        raise ValueError(f"Geometry records missing required columns: {sorted(missing)}")

    config_set = set(config_ids)
    filtered = [row for row in rows if str(row.get("config_id")) in config_set]
    if not filtered:
        raise ValueError("No geometry rows remain after filtering to requested config_ids.")

    validate_episode_split_disjoint(filtered)
    validate_no_duplicate_seed_records(filtered)

    grouped: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in filtered:
        grouped[
            (
                str(row.get("model_slug") or "model"),
                str(row["episode_id"]),
                str(row["config_id"]),
            )
        ].append(row)

    out: List[Dict[str, Any]] = []
    for key, group in sorted(grouped.items()):
        metric_values = {
            metric: [as_float(row.get(metric), field=metric) for row in group]
            for metric in OUTCOME_COLUMNS
        }
        seeds = sorted({int(float(row["seed"])) for row in group})
        record = {
            "model_slug": key[0],
            "episode_id": key[1],
            "learning_type": stable_unique([row.get("learning_type") for row in group], field="learning_type", key=key),
            "meta_split": stable_unique([row.get("meta_split") for row in group], field="meta_split", key=key),
            "model_name": stable_unique([row.get("model_name") for row in group], field="model_name", key=key),
            "config_id": key[2],
            "number_of_seeds": len(seeds),
            "seed_values": "|".join(str(seed) for seed in seeds),
        }
        for metric in OUTCOME_COLUMNS:
            record[metric] = mean(metric_values[metric])
        record["utility"] = mean(record[metric] for metric in OUTCOME_COLUMNS)
        out.append(record)

    diagnostics = coverage_diagnostics(out, config_ids=config_ids)
    if strict:
        validate_strict_coverage(
            out,
            config_ids=config_ids,
            expected_seeds_by_split=expected_seeds_by_split or {},
        )
    return out, diagnostics


def coverage_diagnostics(
    rows: Sequence[Dict[str, Any]],
    config_ids: Sequence[str],
) -> Dict[str, Any]:
    configs = list(config_ids)
    by_episode: Dict[Tuple[str, str, str, str], set[str]] = defaultdict(set)
    for row in rows:
        by_episode[
            (
                str(row.get("model_slug") or "model"),
                str(row["meta_split"]),
                str(row["learning_type"]),
                str(row["episode_id"]),
            )
        ].add(str(row["config_id"]))

    incomplete = []
    complete_by_split: Dict[str, int] = defaultdict(int)
    total_by_split: Dict[str, int] = defaultdict(int)
    for (model_slug, split, learning_type, episode_id), present in sorted(by_episode.items()):
        total_by_split[split] += 1
        missing = [cfg for cfg in configs if cfg not in present]
        if missing:
            incomplete.append(
                {
                    "model_slug": model_slug,
                    "meta_split": split,
                    "learning_type": learning_type,
                    "episode_id": episode_id,
                    "missing_config_ids": missing,
                }
            )
        else:
            complete_by_split[split] += 1

    rows_by_split_config: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    seed_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in rows:
        split = str(row["meta_split"])
        rows_by_split_config[split][str(row["config_id"])] += 1
        seed_counts[split][str(row["number_of_seeds"])] += 1

    return {
        "config_ids": configs,
        "n_rows": len(rows),
        "n_episode_config_units": len(rows),
        "episode_counts_by_split": dict(sorted(total_by_split.items())),
        "complete_episode_counts_by_split": dict(sorted(complete_by_split.items())),
        "rows_by_split_config": {
            split: dict(sorted(counts.items()))
            for split, counts in sorted(rows_by_split_config.items())
        },
        "seed_count_distribution_by_split": {
            split: dict(sorted(counts.items(), key=lambda item: int(item[0])))
            for split, counts in sorted(seed_counts.items())
        },
        "n_incomplete_episodes": len(incomplete),
        "incomplete_episodes": incomplete[:50],
    }


FIELDS = [
    "model_slug",
    "episode_id",
    "learning_type",
    "meta_split",
    "model_name",
    "config_id",
    "acquisition",
    "transfer",
    "boundedness",
    "preservation",
    "utility",
    "number_of_seeds",
    "seed_values",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Experiment 2 geometry labels.")
    parser.add_argument("--records", required=True, help="Geometry records CSV or JSONL.")
    parser.add_argument("--config_ids", nargs="+", default=PRIMARY_CONFIGS)
    parser.add_argument("--expected_train_seeds", nargs="+", type=int, default=[11])
    parser.add_argument("--expected_validation_seeds", nargs="+", type=int, default=[11])
    parser.add_argument("--expected_test_seeds", nargs="+", type=int, default=[11, 22, 33])
    parser.add_argument(
        "--allow_incomplete_smoke_test",
        action="store_true",
        help="Disable strict config/seed coverage checks for tiny pilot fixtures only.",
    )
    parser.add_argument("--output", default="outputs/compiler/experiment2/geometry_dataset.csv")
    parser.add_argument(
        "--diagnostics_output",
        default=None,
        help="Optional diagnostics JSON path. Defaults to <output>.diagnostics.json.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows, diagnostics = prepare_geometry_dataset(
        read_table(args.records),
        config_ids=args.config_ids,
        expected_seeds_by_split=expected_seed_map(
            train=args.expected_train_seeds,
            validation=args.expected_validation_seeds,
            test=args.expected_test_seeds,
        ),
        strict=not args.allow_incomplete_smoke_test,
    )
    output = Path(args.output)
    write_csv(output, rows, FIELDS)

    diagnostics_path = Path(args.diagnostics_output) if args.diagnostics_output else output.with_suffix(".diagnostics.json")
    diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostics_path.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")

    print(f"Wrote {len(rows)} seed-aggregated geometry rows to {output}")
    print(f"Wrote coverage diagnostics to {diagnostics_path}")
    if diagnostics["n_incomplete_episodes"]:
        print(f"[warn] {diagnostics['n_incomplete_episodes']} episode(s) lack at least one requested config.")
    print(json.dumps({k: diagnostics[k] for k in ["episode_counts_by_split", "complete_episode_counts_by_split", "seed_count_distribution_by_split"]}, indent=2))


if __name__ == "__main__":
    main()
