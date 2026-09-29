#!/usr/bin/env python3
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence
import argparse
import random
from collections import defaultdict

try:
    from .common import read_jsonl, write_jsonl
except ImportError:  # pragma: no cover - script execution path
    from common import read_jsonl, write_jsonl

DATA_DIR = Path("data")
BUDGETS_PER_SPEC = [1, 2, 4, 8, 12]
SPECS_PER_TYPE = [10, 25, 50, 100]
SEED = 2026


def _example_ids(rows: Iterable[Dict[str, Any]]) -> List[str]:
    return [row["example_id"] for row in rows]


def build_manifest(
    examples: Sequence[Dict[str, Any]],
    *,
    budgets_per_spec: Sequence[int] = BUDGETS_PER_SPEC,
    specs_per_type: Sequence[int] = SPECS_PER_TYPE,
    seed: int = SEED,
) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    by_type_spec_split = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    spec_ids_by_type = defaultdict(set)

    for e in examples:
        lt, sid = e["learning_type"], e["spec_id"]
        by_type_spec_split[lt][sid][e["split"]].append(e)
        spec_ids_by_type[lt].add(sid)

    manifest = []
    for lt, sids_set in sorted(spec_ids_by_type.items()):
        all_sids = sorted(sids_set)
        rng.shuffle(all_sids)
        for n_specs in specs_per_type:
            if n_specs > len(all_sids):
                continue
            selected_specs = sorted(all_sids[:n_specs])
            for budget in budgets_per_spec:
                train_ids, id_ids, para_ids, gen_ids, neg_ids = [], [], [], [], []
                for sid in selected_specs:
                    train = sorted(by_type_spec_split[lt][sid]["train"], key=lambda x: x["example_id"])[:budget]
                    train_ids.extend(_example_ids(train))
                    id_ids.extend(_example_ids(by_type_spec_split[lt][sid]["id_eval"]))
                    para_ids.extend(_example_ids(by_type_spec_split[lt][sid]["paraphrase_eval"]))
                    gen_ids.extend(_example_ids(by_type_spec_split[lt][sid]["generalization"]))
                    neg_ids.extend(_example_ids(by_type_spec_split[lt][sid]["negative_control"]))
                manifest.append({
                    "run_id": f"calib::{lt}::specs{n_specs}::budget{budget}",
                    "learning_type": lt,
                    "n_specs": n_specs,
                    "train_budget_per_spec": budget,
                    "spec_ids": selected_specs,
                    "train_example_ids": train_ids,
                    "id_eval_example_ids": id_ids,
                    "paraphrase_eval_example_ids": para_ids,
                    "generalization_example_ids": gen_ids,
                    "negative_control_example_ids": neg_ids,
                })
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=str(DATA_DIR))
    parser.add_argument("--output", default=None)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    output = Path(args.output) if args.output else data_dir / "calibration_manifest.jsonl"
    examples = read_jsonl(data_dir / "prompt_examples.jsonl")
    manifest = build_manifest(examples, seed=args.seed)
    write_jsonl(output, manifest)
    print(f"Wrote {len(manifest)} rows")

if __name__ == "__main__":
    main()
