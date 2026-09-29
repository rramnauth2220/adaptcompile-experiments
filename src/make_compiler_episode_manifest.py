#!/usr/bin/env python3
"""Create one-spec adaptation-compiler episode manifests.

This uses the canonical prompt examples and writes new artifacts under
``data/compiler/`` by default. It does not modify the AAAI dataset or
calibration manifest.
"""

from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

try:  # pragma: no cover - exercised when imported as package
    from .compiler_common import OBJECTIVE_BUDGETS, SPLITS, read_jsonl, write_jsonl
except ImportError:  # pragma: no cover - script execution path
    from compiler_common import OBJECTIVE_BUDGETS, SPLITS, read_jsonl, write_jsonl


DEFAULT_META_COUNTS = {
    "train": 80,
    "validation": 20,
    "test": 20,
}


def sort_examples(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            int((row.get("metadata") or {}).get("train_order") or 0),
            str(row.get("example_id")),
        ),
    )


def group_examples(
    examples: Sequence[Dict[str, Any]],
) -> Dict[str, Dict[str, Dict[str, List[Dict[str, Any]]]]]:
    grouped: Dict[str, Dict[str, Dict[str, List[Dict[str, Any]]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for row in examples:
        grouped[row["learning_type"]][row["spec_id"]][row["split"]].append(row)
    for by_spec in grouped.values():
        for by_split in by_spec.values():
            for split, rows in list(by_split.items()):
                by_split[split] = sort_examples(rows)
    return grouped


def parse_budget_overrides(values: Sequence[str] | None) -> Dict[str, int]:
    budgets = dict(OBJECTIVE_BUDGETS)
    for value in values or []:
        if "=" not in value:
            raise ValueError(f"Budget overrides must look like objective=N, got {value!r}")
        objective, budget = value.split("=", 1)
        if objective not in OBJECTIVE_BUDGETS:
            raise ValueError(f"Unknown objective in budget override: {objective}")
        budgets[objective] = int(budget)
    return budgets


def split_spec_ids(
    spec_ids: Sequence[str],
    train_count: int,
    validation_count: int,
    test_count: int,
    seed: int,
) -> Dict[str, List[str]]:
    requested = train_count + validation_count + test_count
    if requested > len(spec_ids):
        raise ValueError(f"Requested {requested} specs but only {len(spec_ids)} are available")
    ordered = sorted(spec_ids)
    rng = random.Random(seed)
    rng.shuffle(ordered)
    return {
        "train": sorted(ordered[:train_count]),
        "validation": sorted(ordered[train_count : train_count + validation_count]),
        "test": sorted(ordered[train_count + validation_count : requested]),
    }


def build_episode_row(
    learning_type: str,
    spec_id: str,
    meta_split: str,
    examples_by_split: Dict[str, List[Dict[str, Any]]],
    budget: int,
) -> Dict[str, Any]:
    train_rows = examples_by_split.get("train", [])
    if budget > len(train_rows):
        raise ValueError(
            f"Budget {budget} exceeds available train examples for {learning_type}/{spec_id}: "
            f"{len(train_rows)}"
        )
    missing = [split for split in SPLITS if split not in examples_by_split]
    if missing:
        raise ValueError(f"{learning_type}/{spec_id} is missing splits: {missing}")

    return {
        "episode_id": f"compiler::{learning_type}::{spec_id}",
        "meta_split": meta_split,
        "learning_type": learning_type,
        "spec_ids": [spec_id],
        "train_budget_per_spec": int(budget),
        "train_example_ids": [row["example_id"] for row in train_rows[:budget]],
        "id_eval_example_ids": [row["example_id"] for row in examples_by_split["id_eval"]],
        "paraphrase_eval_example_ids": [
            row["example_id"] for row in examples_by_split["paraphrase_eval"]
        ],
        "generalization_example_ids": [
            row["example_id"] for row in examples_by_split["generalization"]
        ],
        "negative_control_example_ids": [
            row["example_id"] for row in examples_by_split["negative_control"]
        ],
    }


def build_episode_manifest(
    examples: Sequence[Dict[str, Any]],
    train_count: int = DEFAULT_META_COUNTS["train"],
    validation_count: int = DEFAULT_META_COUNTS["validation"],
    test_count: int = DEFAULT_META_COUNTS["test"],
    split_seed: int = 2026,
    objectives: Sequence[str] | None = None,
    budgets: Dict[str, int] | None = None,
) -> List[Dict[str, Any]]:
    budgets = dict(budgets or OBJECTIVE_BUDGETS)
    grouped = group_examples(examples)
    selected_objectives = list(objectives or sorted(grouped))

    rows: List[Dict[str, Any]] = []
    for objective in selected_objectives:
        if objective not in grouped:
            raise ValueError(f"No examples found for objective {objective!r}")
        if objective not in budgets:
            raise ValueError(f"No budget configured for objective {objective!r}")
        spec_ids = sorted(grouped[objective])
        splits = split_spec_ids(
            spec_ids,
            train_count=train_count,
            validation_count=validation_count,
            test_count=test_count,
            seed=split_seed + sum(ord(c) for c in objective),
        )
        for meta_split in ["train", "validation", "test"]:
            for spec_id in splits[meta_split]:
                rows.append(
                    build_episode_row(
                        learning_type=objective,
                        spec_id=spec_id,
                        meta_split=meta_split,
                        examples_by_split=grouped[objective][spec_id],
                        budget=budgets[objective],
                    )
                )

    return sorted(rows, key=lambda row: (row["learning_type"], row["meta_split"], row["spec_ids"][0]))


def validate_episode_manifest(
    rows: Sequence[Dict[str, Any]],
    examples: Sequence[Dict[str, Any]],
    train_count: int | None = None,
    validation_count: int | None = None,
    test_count: int | None = None,
) -> None:
    examples_by_id = {row["example_id"]: row for row in examples}
    spec_to_split: Dict[str, str] = {}
    counts: Dict[Tuple[str, str], int] = defaultdict(int)

    for row in rows:
        required = {
            "episode_id",
            "meta_split",
            "learning_type",
            "spec_ids",
            "train_budget_per_spec",
            "train_example_ids",
            "id_eval_example_ids",
            "paraphrase_eval_example_ids",
            "generalization_example_ids",
            "negative_control_example_ids",
        }
        missing = required - set(row)
        if missing:
            raise ValueError(f"Episode row missing fields: {sorted(missing)}")
        if len(row["spec_ids"]) != 1:
            raise ValueError(f"Initial compiler manifests require exactly one spec: {row['episode_id']}")

        spec_id = row["spec_ids"][0]
        meta_split = row["meta_split"]
        learning_type = row["learning_type"]
        if spec_id in spec_to_split and spec_to_split[spec_id] != meta_split:
            raise ValueError(f"Spec leakage across meta splits: {spec_id}")
        spec_to_split[spec_id] = meta_split
        counts[(learning_type, meta_split)] += 1

        split_keys = {
            "train_example_ids": "train",
            "id_eval_example_ids": "id_eval",
            "paraphrase_eval_example_ids": "paraphrase_eval",
            "generalization_example_ids": "generalization",
            "negative_control_example_ids": "negative_control",
        }
        for key, expected_split in split_keys.items():
            ids = row[key]
            if len(ids) != len(set(ids)):
                raise ValueError(f"Duplicate example IDs in {row['episode_id']} {key}")
            for example_id in ids:
                if example_id not in examples_by_id:
                    raise ValueError(f"{row['episode_id']} references missing example {example_id}")
                example = examples_by_id[example_id]
                if example["spec_id"] != spec_id:
                    raise ValueError(f"{row['episode_id']} mixes spec IDs via {example_id}")
                if example["split"] != expected_split:
                    raise ValueError(f"{row['episode_id']} {example_id} is not split {expected_split}")
                if example["learning_type"] != learning_type:
                    raise ValueError(f"{row['episode_id']} mixes learning types via {example_id}")

    expected_counts = {
        "train": train_count,
        "validation": validation_count,
        "test": test_count,
    }
    if any(v is not None for v in expected_counts.values()):
        objectives = sorted({row["learning_type"] for row in rows})
        for objective in objectives:
            for split, expected in expected_counts.items():
                if expected is not None and counts[(objective, split)] != expected:
                    raise ValueError(
                        f"{objective}/{split} has {counts[(objective, split)]} specs; expected {expected}"
                    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build adaptation-compiler episode manifest.")
    parser.add_argument("--examples_path", default="data/prompt_examples.jsonl")
    parser.add_argument("--output", default="data/compiler/episode_manifest.jsonl")
    parser.add_argument("--objectives", nargs="+", default=list(OBJECTIVE_BUDGETS), choices=list(OBJECTIVE_BUDGETS))
    parser.add_argument("--train_count", type=int, default=DEFAULT_META_COUNTS["train"])
    parser.add_argument("--validation_count", type=int, default=DEFAULT_META_COUNTS["validation"])
    parser.add_argument("--test_count", type=int, default=DEFAULT_META_COUNTS["test"])
    parser.add_argument("--split_seed", type=int, default=2026)
    parser.add_argument("--budget", nargs="*", default=[], help="Optional objective=N overrides.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    examples = read_jsonl(args.examples_path)
    budgets = parse_budget_overrides(args.budget)
    rows = build_episode_manifest(
        examples,
        train_count=args.train_count,
        validation_count=args.validation_count,
        test_count=args.test_count,
        split_seed=args.split_seed,
        objectives=args.objectives,
        budgets=budgets,
    )
    validate_episode_manifest(
        rows,
        examples,
        train_count=args.train_count,
        validation_count=args.validation_count,
        test_count=args.test_count,
    )
    write_jsonl(args.output, rows)
    print(f"Wrote {len(rows)} compiler episodes to {args.output}")


if __name__ == "__main__":
    main()
