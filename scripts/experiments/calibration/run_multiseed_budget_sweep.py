#!/usr/bin/env python3
"""
Run a multi-seed candidate-budget sweep.

This is intended as the second-stage confirmation sweep after a coarse
single-seed sweep has identified plausible budgets.

Example:

  python scripts/experiments/calibration/run_multiseed_budget_sweep.py \
    --learning_type lexical_binding \
    --n_specs 25 \
    --budgets 6 8 10 \
    --seeds 11 22 33 \
    --run_root outputs/multiseed_lexical \
    --results_root outputs/multiseed_lexical

  python scripts/experiments/calibration/run_multiseed_budget_sweep.py \
    --learning_type factual_association \
    --n_specs 25 \
    --budgets 8 10 12 \
    --seeds 11 22 33 \
    --run_root outputs/multiseed_factual \
    --results_root outputs/multiseed_factual

Notes:
  - Baseline evaluation is run once per budget.
  - Adapter training/evaluation is run once per budget x seed.
  - The canonical training script exposes --seed for reproducible sweeps.
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument("--model_name", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--learning_type", required=True)
    p.add_argument("--n_specs", type=int, default=25)
    p.add_argument("--budgets", nargs="+", type=int, required=True)
    p.add_argument("--seeds", nargs="+", type=int, default=[11, 22, 33])

    p.add_argument("--run_root", required=True)
    p.add_argument("--results_root", required=True)

    p.add_argument("--num_train_epochs", type=float, default=3)
    p.add_argument("--max_new_tokens", type=int, default=32)
    p.add_argument("--learning_rate", default="2e-4")
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--target_modules", nargs="+", default=DEFAULT_TARGET_MODULES)
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)

    p.add_argument("--train_script", default="src/train_fullstack_lora.py")
    p.add_argument("--eval_script", default="src/evaluate_calibration_run.py")
    p.add_argument("--summary_script", default="src/summarize_results.py")

    p.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip a training/eval step if its expected output already exists.",
    )
    p.add_argument(
        "--allow_missing_seed_arg",
        action="store_true",
        help=(
            "Allow running even if train_fullstack_lora.py does not expose --seed. "
            "This is not recommended for true multi-seed confirmation."
        ),
    )
    return p.parse_args()


def run(cmd: list[str | Path | int | float]) -> None:
    cmd = [str(x) for x in cmd]
    print("\n$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def script_supports_arg(script_path: str | Path, arg_name: str) -> bool:
    try:
        result = subprocess.run(
            ["python", str(script_path), "--help"],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
    except subprocess.CalledProcessError as e:
        text = e.stdout or ""
    else:
        text = result.stdout or ""
    return arg_name in text


def main() -> None:
    args = parse_args()

    run_root = Path(args.run_root)
    results_root = Path(args.results_root)
    run_root.mkdir(parents=True, exist_ok=True)
    results_root.mkdir(parents=True, exist_ok=True)

    train_has_seed = script_supports_arg(args.train_script, "--seed")
    if not train_has_seed and not args.allow_missing_seed_arg:
        raise SystemExit(
            "\nERROR: src/train_fullstack_lora.py does not appear to support --seed.\n"
            "Use a training script that exposes --seed, then rerun this command.\n"
        )

    print("Multi-seed budget sweep")
    print("-----------------------")
    print(f"model_name:      {args.model_name}")
    print(f"learning_type:   {args.learning_type}")
    print(f"n_specs:         {args.n_specs}")
    print(f"budgets:         {args.budgets}")
    print(f"seeds:           {args.seeds}")
    print(f"target_modules:  {args.target_modules}")
    print(f"train_has_seed:  {train_has_seed}")
    print(f"run_root:        {run_root}")
    print(f"results_root:    {results_root}")

    all_jsonl: list[str] = []

    for budget in args.budgets:
        run_id = f"calib::{args.learning_type}::specs{args.n_specs}::budget{budget}"
        safe_base = f"calib_{args.learning_type}_specs{args.n_specs}_budget{budget}"

        baseline_jsonl = results_root / f"baseline_{safe_base}.jsonl"

        print("\n" + "=" * 80)
        print(f"Budget {budget}: {run_id}")
        print("=" * 80)

        if args.skip_existing and baseline_jsonl.exists():
            print(f"Skipping existing baseline: {baseline_jsonl}")
        else:
            run([
                "python", args.eval_script,
                "--model_name", args.model_name,
                "--run_id", run_id,
                "--output_path", baseline_jsonl,
                "--max_new_tokens", args.max_new_tokens,
            ])

        all_jsonl.append(str(baseline_jsonl))

        for seed in args.seeds:
            safe_run = f"{safe_base}_seed{seed}"
            out_dir = run_root / safe_run
            adapter_jsonl = results_root / f"adapter_{safe_run}.jsonl"

            print("\n" + "-" * 80)
            print(f"Budget {budget}, seed {seed}")
            print("-" * 80)

            if args.skip_existing and adapter_jsonl.exists():
                print(f"Skipping existing adapter eval: {adapter_jsonl}")
                all_jsonl.append(str(adapter_jsonl))
                continue

            train_cmd = [
                "python", args.train_script,
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
            if train_has_seed:
                train_cmd.extend(["--seed", seed])

            run(train_cmd)

            run([
                "python", args.eval_script,
                "--model_name", args.model_name,
                "--adapter_path", out_dir,
                "--run_id", run_id,
                "--output_path", adapter_jsonl,
                "--max_new_tokens", args.max_new_tokens,
            ])

            all_jsonl.append(str(adapter_jsonl))

    combined_summary = results_root / f"summary_{args.learning_type}_specs{args.n_specs}_multiseed.csv"

    run([
        "python", args.summary_script,
        "--inputs", *all_jsonl,
        "--output_csv", combined_summary,
        "--figure_dir", results_root / "figures",
        "--figure_prefix", f"{args.learning_type}_specs{args.n_specs}_multiseed",
    ])

    print("\nDone.")
    print(f"Raw JSONL files: {results_root}")
    print(f"Adapters:        {run_root}")
    print(f"Combined CSV:    {combined_summary}")
    print()
    print("Next, select a budget with:")
    print(
        "  python scripts/experiments/calibration/select_budget_bootstrap.py "
        f"--inputs {results_root}/adapter_*.jsonl "
        f"--output_csv {results_root}/budget_selection_bootstrap.csv "
        f"--decision_json {results_root}/budget_selection_decision.json "
        f"--figure_dir {results_root}/figures_bootstrap"
    )


if __name__ == "__main__":
    main()
