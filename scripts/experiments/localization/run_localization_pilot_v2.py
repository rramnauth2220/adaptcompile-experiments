#!/usr/bin/env python3
"""
Run a localization pilot using the patched src/train_fullstack_lora.py.

Unlike the earlier pilot runner, this uses the same trainer as calibration.
The only intended difference between conditions is --localization_condition.

Example:

  mkdir -p outputs/llama31_8b_localization_lexical_seed11_v2

  python scripts/experiments/localization/run_localization_pilot_v2.py \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --learning_type lexical_binding \
    --n_specs 25 \
    --budget 10 \
    --seed 11 \
    --conditions full early middle late \
    --run_root outputs/llama31_8b_localization_lexical_seed11_v2 \
    --results_root outputs/llama31_8b_localization_lexical_seed11_v2 \
    --target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
    --num_train_epochs 3 \
    --lora_r 16 \
    --lora_alpha 32 \
    --torch_dtype bfloat16 \
    2>&1 | tee outputs/llama31_8b_localization_lexical_seed11_v2/run.log
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List


DEFAULT_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str | Path, rows: List[Dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def run(cmd: List[Any], dry_run: bool = False) -> None:
    cmd = [str(x) for x in cmd]
    print("\n$", " ".join(cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def enrich_result_jsonl(
    path: str | Path,
    condition: str,
    seed: int,
    adapter_path: str | None,
) -> None:
    rows = read_jsonl(path)
    out_rows = []
    for row in rows:
        out = dict(row)
        # Keep original adapter/baseline field if present but add explicit localization labels.
        out["localization_condition"] = condition
        out["localization_seed"] = seed
        out["condition"] = condition
        if adapter_path is not None:
            out["adapter_path"] = adapter_path
        out_rows.append(out)
    write_jsonl(path, out_rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument("--model_name", required=True)
    p.add_argument("--learning_type", required=True)
    p.add_argument("--n_specs", type=int, default=25)
    p.add_argument("--budget", type=int, required=True)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--conditions", nargs="+", default=["full", "early", "middle", "late"])

    p.add_argument("--run_root", required=True)
    p.add_argument("--results_root", required=True)

    p.add_argument("--train_script", default="src/train_fullstack_lora.py")
    p.add_argument("--eval_script", default="src/evaluate_calibration_run.py")

    p.add_argument("--target_modules", nargs="+", default=DEFAULT_TARGET_MODULES)
    p.add_argument("--region_width", type=int, default=8)
    p.add_argument("--num_train_epochs", type=float, default=3)
    p.add_argument("--max_new_tokens", type=int, default=32)
    p.add_argument("--learning_rate", default="2e-4")
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--torch_dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    p.add_argument("--device_map", default="auto")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--gradient_checkpointing", action="store_true")

    p.add_argument("--skip_existing", action="store_true")
    p.add_argument("--dry_run", action="store_true")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    run_root = Path(args.run_root)
    results_root = Path(args.results_root)
    run_root.mkdir(parents=True, exist_ok=True)
    results_root.mkdir(parents=True, exist_ok=True)

    run_id = f"calib::{args.learning_type}::specs{args.n_specs}::budget{args.budget}"

    for condition in args.conditions:
        safe = f"{args.learning_type}_specs{args.n_specs}_budget{args.budget}_{condition}_seed{args.seed}"

        adapter_dir = run_root / safe
        result_jsonl = results_root / f"adapter_{safe}.jsonl"

        if args.skip_existing and result_jsonl.exists():
            print(f"Skipping existing result: {result_jsonl}")
            continue

        train_cmd = [
            sys.executable, args.train_script,
            "--model_name", args.model_name,
            "--run_id", run_id,
            "--output_dir", adapter_dir,
            "--target_modules", *args.target_modules,
            "--num_train_epochs", args.num_train_epochs,
            "--learning_rate", args.learning_rate,
            "--lora_r", args.lora_r,
            "--lora_alpha", args.lora_alpha,
            "--lora_dropout", args.lora_dropout,
            "--per_device_train_batch_size", args.per_device_train_batch_size,
            "--gradient_accumulation_steps", args.gradient_accumulation_steps,
            "--max_length", args.max_length,
            "--seed", args.seed,
            "--torch_dtype", args.torch_dtype,
            "--device_map", args.device_map,
            "--localization_condition", condition,
            "--region_width", args.region_width,
        ]
        if args.bf16:
            train_cmd.append("--bf16")
        if args.fp16:
            train_cmd.append("--fp16")
        if args.gradient_checkpointing:
            train_cmd.append("--gradient_checkpointing")

        run(train_cmd, dry_run=args.dry_run)

        eval_cmd = [
            sys.executable, args.eval_script,
            "--model_name", args.model_name,
            "--adapter_path", adapter_dir,
            "--run_id", run_id,
            "--output_path", result_jsonl,
            "--max_new_tokens", args.max_new_tokens,
            "--device_map", args.device_map,
            "--torch_dtype", args.torch_dtype,
        ]

        run(eval_cmd, dry_run=args.dry_run)

        if not args.dry_run:
            enrich_result_jsonl(
                result_jsonl,
                condition=condition,
                seed=args.seed,
                adapter_path=str(adapter_dir),
            )

    print(f"\nDone. Results written to {results_root}")


if __name__ == "__main__":
    main()
