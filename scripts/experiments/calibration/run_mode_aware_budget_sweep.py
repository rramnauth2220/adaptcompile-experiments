#!/usr/bin/env python3
"""
Python launcher for mode-aware budget sweeps.

Fix included:
  train_fullstack_lora.py requires --target_modules, so this launcher now passes
  a model-appropriate default list for Qwen/Llama-style transformers.

Usage:
  python scripts/experiments/calibration/run_mode_aware_budget_sweep.py

Custom target modules:
  python scripts/experiments/calibration/run_mode_aware_budget_sweep.py \
    --target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


DEFAULT_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--learning_type", default="lexical_binding")
    p.add_argument("--n_specs", type=int, default=25)
    p.add_argument("--budgets", nargs="+", type=int, default=[1, 2, 4, 8, 12])
    p.add_argument("--num_train_epochs", type=float, default=3)
    p.add_argument("--max_new_tokens", type=int, default=32)

    p.add_argument("--run_root", default="outputs/mode_aware_budget_sweep")
    p.add_argument("--results_root", default="outputs/mode_aware_budget_sweep")

    p.add_argument("--learning_rate", default="2e-4")
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--target_modules", nargs="+", default=DEFAULT_TARGET_MODULES)

    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    return p.parse_args()


def run(cmd: list[str | Path | int | float]) -> None:
    print("\n$", " ".join(map(str, cmd)), flush=True)
    subprocess.run([str(x) for x in cmd], check=True)


def main():
    args = parse_args()

    run_root = Path(args.run_root)
    results_root = Path(args.results_root)
    run_root.mkdir(parents=True, exist_ok=True)
    results_root.mkdir(parents=True, exist_ok=True)

    print("Mode-aware budget sweep")
    print("-----------------------")
    print(f"model_name:      {args.model_name}")
    print(f"learning_type:   {args.learning_type}")
    print(f"n_specs:         {args.n_specs}")
    print(f"budgets:         {args.budgets}")
    print(f"epochs:          {args.num_train_epochs}")
    print(f"target_modules:  {args.target_modules}")
    print(f"run_root:        {run_root}")
    print(f"results_root:    {results_root}")

    all_jsonl: list[str] = []

    for budget in args.budgets:
        run_id = f"calib::{args.learning_type}::specs{args.n_specs}::budget{budget}"
        safe_run = f"calib_{args.learning_type}_specs{args.n_specs}_budget{budget}"

        out_dir = run_root / safe_run
        baseline_jsonl = results_root / f"baseline_{safe_run}.jsonl"
        adapter_jsonl = results_root / f"adapter_{safe_run}.jsonl"
        summary_csv = results_root / f"summary_{safe_run}.csv"

        print("\n" + "=" * 72)
        print(f"Running {run_id}")
        print("=" * 72)

        run([
            "python", "src/evaluate_calibration_run.py",
            "--model_name", args.model_name,
            "--run_id", run_id,
            "--output_path", baseline_jsonl,
            "--max_new_tokens", args.max_new_tokens,
        ])

        train_cmd = [
            "python", "src/train_fullstack_lora.py",
            "--model_name", args.model_name,
            "--run_id", run_id,
            "--output_dir", out_dir,
            "--target_modules", *args.target_modules,
            "--num_train_epochs", args.num_train_epochs,
            "--learning_rate", args.learning_rate,
            "--lora_r", args.lora_r,
            "--lora_alpha", args.lora_alpha,
            "--lora_dropout", args.lora_dropout,
            "--per_device_train_batch_size", args.per_device_train_batch_size,
            "--gradient_accumulation_steps", args.gradient_accumulation_steps,
        ]
        run(train_cmd)

        run([
            "python", "src/evaluate_calibration_run.py",
            "--model_name", args.model_name,
            "--adapter_path", out_dir,
            "--run_id", run_id,
            "--output_path", adapter_jsonl,
            "--max_new_tokens", args.max_new_tokens,
        ])

        run([
            "python", "src/summarize_results.py",
            "--inputs", baseline_jsonl, adapter_jsonl,
            "--output_csv", summary_csv,
        ])

        all_jsonl.extend([str(baseline_jsonl), str(adapter_jsonl)])

    combined_summary = results_root / f"summary_{args.learning_type}_specs{args.n_specs}_all_budgets.csv"
    run([
        "python", "src/summarize_results.py",
        "--inputs", *all_jsonl,
        "--output_csv", combined_summary,
    ])

    print("\nDone.")
    print(f"Raw JSONL files: {results_root}")
    print(f"Adapters:        {run_root}")
    print(f"Combined CSV:    {combined_summary}")


if __name__ == "__main__":
    main()
