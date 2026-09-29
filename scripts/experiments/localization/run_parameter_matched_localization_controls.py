#!/usr/bin/env python3
"""
Run total-parameter-matched localization controls.

The main localization experiments compare full, early, middle, and late LoRA
adaptation. With a 32-layer model and region_width=8, full adaptation touches
roughly four times as many layers as a localized window. This launcher creates
two rank controls that approximately match total trainable capacity by changing
LoRA rank while keeping the selected data budgets fixed.

Outputs are written under:
  outputs/parameter_matched_localization/<control>/<objective>/budget_<B>/
    condition_<condition>/rank_<rank>/seed_<seed>/

Each run directory contains the adapter, raw evaluation JSONL, a per-run
localization summary, and a completion marker. A top-level manifest and
cross-run summaries are also written.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, Iterable, List, Sequence


OBJECTIVE_BUDGETS = {
    "lexical_binding": 10,
    "factual_association": 8,
    "behavioral_policy": 10,
    "causal_mapping": 10,
    "procedural_reasoning": 8,
}

OBJECTIVE_EVAL_SCRIPTS = {
    "lexical_binding": "src/evaluate_calibration_run.py",
    "factual_association": "src/evaluate_calibration_run.py",
    "behavioral_policy": "src/evaluate_calibration_run_behavioral.py",
    "causal_mapping": "src/evaluate_calibration_run_causal.py",
    "procedural_reasoning": "src/evaluate_calibration_run_procedural.py",
}

BOUND_SOURCE = {
    "lexical_binding": "strict_accuracy",
    "factual_association": "strict_accuracy",
    "behavioral_policy": "concept_accuracy",
    "causal_mapping": "concept_accuracy",
    "procedural_reasoning": "concept_accuracy",
}

CONDITIONS = ["full", "early", "middle", "late"]
DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


@dataclass(frozen=True)
class RunSpec:
    control: str
    objective: str
    budget: int
    condition: str
    seed: int
    rank: int
    n_layers_adapted: int
    approximate_rank_layer_product: int
    output_dir: Path
    adapter_dir: Path
    result_jsonl: Path
    per_run_summary_csv: Path
    train_command: List[str]
    eval_command: List[str]
    summarize_command: List[str]

    @property
    def run_id(self) -> str:
        return f"calib::{self.objective}::specs25::budget{self.budget}"

    @property
    def manifest_command(self) -> str:
        return " && ".join(
            [
                shell_join(self.train_command),
                shell_join(self.eval_command),
                shell_join(self.summarize_command),
            ]
        )


def shell_join(cmd: Sequence[Any]) -> str:
    parts = [str(x) for x in cmd]
    if os.name == "nt":
        return subprocess.list2cmdline(parts)
    return shlex.join(parts)


def run_command(cmd: Sequence[Any], dry_run: bool) -> None:
    printable = shell_join(cmd)
    print(f"\n$ {printable}", flush=True)
    if dry_run:
        return
    subprocess.run([str(x) for x in cmd], check=True)


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_csv(path: str | Path) -> List[Dict[str, str]]:
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: str | Path, rows: List[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def maybe_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def sd(values: List[float]) -> float | None:
    return stdev(values) if len(values) > 1 else None


def rank_for(control: str, condition: str, base_rank: int, rank_multiplier: int) -> int:
    if control == "localized_expanded_rank":
        return base_rank if condition == "full" else base_rank * rank_multiplier
    if control == "full_reduced_rank":
        return max(1, base_rank // rank_multiplier) if condition == "full" else base_rank
    raise ValueError(f"Unknown control: {control}")


def is_complete(spec: RunSpec) -> bool:
    return (
        (spec.output_dir / ".complete").exists()
        and spec.result_jsonl.exists()
        and spec.per_run_summary_csv.exists()
    )


def enrich_result_jsonl(spec: RunSpec) -> None:
    rows = []
    for row in read_jsonl(spec.result_jsonl):
        out = dict(row)
        out["parameter_matched_control"] = spec.control
        out["localization_condition"] = spec.condition
        out["condition"] = spec.condition
        out["seed"] = spec.seed
        out["localization_seed"] = spec.seed
        out["rank"] = spec.rank
        out["lora_r"] = spec.rank
        out["n_layers_adapted"] = spec.n_layers_adapted
        out["approximate_rank_layer_product"] = spec.approximate_rank_layer_product
        out["adapter_path"] = str(spec.adapter_dir)
        rows.append(out)
    write_jsonl(spec.result_jsonl, rows)


def load_trainable_parameters(adapter_dir: Path) -> int | None:
    metadata_path = adapter_dir / "localization_metadata.json"
    if not metadata_path.exists():
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    value = metadata.get("trainable_parameters")
    return int(value) if value is not None else None


def build_specs(args: argparse.Namespace) -> List[RunSpec]:
    rank_multiplier = args.n_layers / args.region_width
    if abs(rank_multiplier - round(rank_multiplier)) > 1e-9:
        print(
            f"[warn] n_layers / region_width = {rank_multiplier:.3f}; "
            "using rounded rank multiplier for approximate matching."
        )
    rank_multiplier_int = max(1, round(rank_multiplier))

    specs: List[RunSpec] = []
    for control in args.controls:
        for objective in args.objectives:
            budget = OBJECTIVE_BUDGETS[objective]
            run_id = f"calib::{objective}::specs25::budget{budget}"
            eval_script = OBJECTIVE_EVAL_SCRIPTS[objective]

            for condition in CONDITIONS:
                n_layers_adapted = args.n_layers if condition == "full" else args.region_width
                rank = rank_for(control, condition, args.base_rank, rank_multiplier_int)
                approx_product = rank * n_layers_adapted

                for seed in args.seeds:
                    output_dir = (
                        args.output_root
                        / control
                        / objective
                        / f"budget_{budget}"
                        / f"condition_{condition}"
                        / f"rank_{rank}"
                        / f"seed_{seed}"
                    )
                    adapter_dir = output_dir / "adapter"
                    safe = f"{objective}_specs25_budget{budget}_{condition}_rank{rank}_seed{seed}"
                    result_jsonl = output_dir / f"adapter_{safe}.jsonl"
                    per_run_summary_csv = output_dir / "summary_localization_by_seed.csv"

                    train_cmd: List[Any] = [
                        args.python_exe,
                        args.train_script,
                        "--model_name",
                        args.model_name,
                        "--run_id",
                        run_id,
                        "--output_dir",
                        adapter_dir,
                        "--target_modules",
                        *args.target_modules,
                        "--num_train_epochs",
                        args.num_train_epochs,
                        "--learning_rate",
                        args.learning_rate,
                        "--lora_r",
                        rank,
                        "--lora_alpha",
                        args.lora_alpha,
                        "--lora_dropout",
                        args.lora_dropout,
                        "--per_device_train_batch_size",
                        args.per_device_train_batch_size,
                        "--gradient_accumulation_steps",
                        args.gradient_accumulation_steps,
                        "--max_length",
                        args.max_length,
                        "--seed",
                        seed,
                        "--torch_dtype",
                        args.torch_dtype,
                        "--device_map",
                        args.device_map,
                        "--localization_condition",
                        condition,
                        "--region_width",
                        args.region_width,
                    ]
                    if args.bf16:
                        train_cmd.append("--bf16")
                    if args.fp16:
                        train_cmd.append("--fp16")
                    if args.gradient_checkpointing:
                        train_cmd.append("--gradient_checkpointing")

                    eval_cmd: List[Any] = [
                        args.python_exe,
                        eval_script,
                        "--model_name",
                        args.model_name,
                        "--adapter_path",
                        adapter_dir,
                        "--run_id",
                        run_id,
                        "--output_path",
                        result_jsonl,
                        "--max_new_tokens",
                        args.max_new_tokens,
                        "--device_map",
                        args.device_map,
                        "--torch_dtype",
                        args.torch_dtype,
                    ]

                    summarize_cmd: List[Any] = [
                        args.python_exe,
                        args.summarize_script,
                        "--inputs",
                        result_jsonl,
                        "--output_csv",
                        per_run_summary_csv,
                    ]

                    specs.append(
                        RunSpec(
                            control=control,
                            objective=objective,
                            budget=budget,
                            condition=condition,
                            seed=seed,
                            rank=rank,
                            n_layers_adapted=n_layers_adapted,
                            approximate_rank_layer_product=approx_product,
                            output_dir=output_dir,
                            adapter_dir=adapter_dir,
                            result_jsonl=result_jsonl,
                            per_run_summary_csv=per_run_summary_csv,
                            train_command=[str(x) for x in train_cmd],
                            eval_command=[str(x) for x in eval_cmd],
                            summarize_command=[str(x) for x in summarize_cmd],
                        )
                    )
    return specs


def discover_completed_specs(
    output_root: Path,
    controls: Sequence[str],
    n_layers: int,
    region_width: int,
) -> List[RunSpec]:
    specs: List[RunSpec] = []
    allowed_controls = set(controls)

    for summary_csv in sorted(output_root.glob("*/*/budget_*/condition_*/rank_*/seed_*/summary_localization_by_seed.csv")):
        output_dir = summary_csv.parent
        try:
            rel = output_dir.relative_to(output_root)
            control = rel.parts[0]
            objective = rel.parts[1]
            budget = int(rel.parts[2].removeprefix("budget_"))
            condition = rel.parts[3].removeprefix("condition_")
            rank = int(rel.parts[4].removeprefix("rank_"))
            seed = int(rel.parts[5].removeprefix("seed_"))
        except (IndexError, ValueError):
            continue

        if control not in allowed_controls:
            continue
        if objective not in OBJECTIVE_BUDGETS:
            continue
        if condition not in CONDITIONS:
            continue

        n_layers_adapted = n_layers if condition == "full" else region_width
        result_candidates = sorted(output_dir.glob("adapter_*.jsonl")) + sorted(output_dir.glob("adapter_*.jsonl.gz"))
        result_jsonl = result_candidates[0] if result_candidates else output_dir / "adapter_missing.jsonl"

        specs.append(
            RunSpec(
                control=control,
                objective=objective,
                budget=budget,
                condition=condition,
                seed=seed,
                rank=rank,
                n_layers_adapted=n_layers_adapted,
                approximate_rank_layer_product=rank * n_layers_adapted,
                output_dir=output_dir,
                adapter_dir=output_dir / "adapter",
                result_jsonl=result_jsonl,
                per_run_summary_csv=summary_csv,
                train_command=[],
                eval_command=[],
                summarize_command=[],
            )
        )

    return specs


def write_manifest(specs: Sequence[RunSpec], path: Path) -> None:
    rows = [
        {
            "control": spec.control,
            "objective": spec.objective,
            "budget": spec.budget,
            "condition": spec.condition,
            "seed": spec.seed,
            "rank": spec.rank,
            "n_layers_adapted": spec.n_layers_adapted,
            "approximate_rank_layer_product": spec.approximate_rank_layer_product,
            "output_dir": str(spec.output_dir),
            "command": spec.manifest_command,
        }
        for spec in specs
    ]
    write_csv(
        path,
        rows,
        [
            "control",
            "objective",
            "budget",
            "condition",
            "seed",
            "rank",
            "n_layers_adapted",
            "approximate_rank_layer_product",
            "output_dir",
            "command",
        ],
    )


def split_metric(rows: List[Dict[str, str]], split: str, metric: str) -> float | None:
    for row in rows:
        if (
            row.get("split") == split
            and row.get("example_mode") == "ALL_MODES"
            and row.get("scoring_type") == "ALL_SCORING_TYPES"
        ):
            return maybe_float(row.get(metric))
    return None


def summarize_completed_runs(specs: Sequence[RunSpec], output_root: Path) -> None:
    by_seed_rows: List[Dict[str, Any]] = []

    for spec in specs:
        if not spec.per_run_summary_csv.exists():
            continue

        rows = read_csv(spec.per_run_summary_csv)
        id_eval = split_metric(rows, "id_eval", "strict_accuracy")
        paraphrase = split_metric(rows, "paraphrase_eval", "strict_accuracy")
        transfer = split_metric(rows, "generalization", "strict_accuracy")
        boundedness = split_metric(rows, "negative_control", BOUND_SOURCE[spec.objective])

        if id_eval is None or paraphrase is None or transfer is None or boundedness is None:
            print(f"[warn] Could not extract all metrics from {spec.per_run_summary_csv}")
            continue

        trainable_parameters = load_trainable_parameters(spec.adapter_dir)
        by_seed_rows.append(
            {
                "control": spec.control,
                "objective": spec.objective,
                "budget": spec.budget,
                "condition": spec.condition,
                "seed": spec.seed,
                "rank": spec.rank,
                "n_layers_adapted": spec.n_layers_adapted,
                "approximate_rank_layer_product": spec.approximate_rank_layer_product,
                "trainable_parameters": trainable_parameters,
                "id_eval": id_eval,
                "paraphrase_eval": paraphrase,
                "acquisition": (id_eval + paraphrase) / 2,
                "transfer": transfer,
                "generalization": transfer,
                "boundedness": boundedness,
                "output_dir": str(spec.output_dir),
                "summary_file": str(spec.per_run_summary_csv),
            }
        )

    by_seed_path = output_root / "summary_parameter_matched_by_seed.csv"
    write_csv(
        by_seed_path,
        by_seed_rows,
        [
            "control",
            "objective",
            "budget",
            "condition",
            "seed",
            "rank",
            "n_layers_adapted",
            "approximate_rank_layer_product",
            "trainable_parameters",
            "id_eval",
            "paraphrase_eval",
            "acquisition",
            "transfer",
            "generalization",
            "boundedness",
            "output_dir",
            "summary_file",
        ],
    )

    grouped: Dict[tuple, List[Dict[str, Any]]] = {}
    for row in by_seed_rows:
        key = (
            row["control"],
            row["objective"],
            row["budget"],
            row["condition"],
            row["rank"],
            row["n_layers_adapted"],
            row["approximate_rank_layer_product"],
        )
        grouped.setdefault(key, []).append(row)

    aggregate_rows: List[Dict[str, Any]] = []
    for key, rows in sorted(grouped.items()):
        (
            control,
            objective,
            budget,
            condition,
            rank,
            n_layers_adapted,
            approx_product,
        ) = key

        def values(metric: str) -> List[float]:
            return [float(row[metric]) for row in rows if row[metric] not in (None, "")]

        trainable_values = [
            int(row["trainable_parameters"])
            for row in rows
            if row["trainable_parameters"] not in (None, "")
        ]

        acquisition = values("acquisition")
        transfer = values("transfer")
        boundedness = values("boundedness")

        aggregate_rows.append(
            {
                "control": control,
                "objective": objective,
                "budget": budget,
                "condition": condition,
                "rank": rank,
                "n_layers_adapted": n_layers_adapted,
                "approximate_rank_layer_product": approx_product,
                "trainable_parameters_mean": mean(trainable_values) if trainable_values else None,
                "acquisition_mean": mean(acquisition) if acquisition else None,
                "acquisition_sd": sd(acquisition),
                "transfer_mean": mean(transfer) if transfer else None,
                "transfer_sd": sd(transfer),
                "generalization_mean": mean(transfer) if transfer else None,
                "generalization_sd": sd(transfer),
                "boundedness_mean": mean(boundedness) if boundedness else None,
                "boundedness_sd": sd(boundedness),
                "n_seeds": len({row["seed"] for row in rows}),
            }
        )

    aggregate_path = output_root / "summary_parameter_matched_aggregate.csv"
    write_csv(
        aggregate_path,
        aggregate_rows,
        [
            "control",
            "objective",
            "budget",
            "condition",
            "rank",
            "n_layers_adapted",
            "approximate_rank_layer_product",
            "trainable_parameters_mean",
            "acquisition_mean",
            "acquisition_sd",
            "transfer_mean",
            "transfer_sd",
            "generalization_mean",
            "generalization_sd",
            "boundedness_mean",
            "boundedness_sd",
            "n_seeds",
        ],
    )

    print(f"Wrote {by_seed_path}")
    print(f"Wrote {aggregate_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run total-parameter-matched controls for localization experiments."
    )
    parser.add_argument(
        "--objectives",
        nargs="+",
        default=list(OBJECTIVE_BUDGETS),
        choices=list(OBJECTIVE_BUDGETS),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 22, 33])
    parser.add_argument("--base_rank", type=int, default=8)
    parser.add_argument(
        "--controls",
        nargs="+",
        default=["localized_expanded_rank", "full_reduced_rank"],
        choices=["localized_expanded_rank", "full_reduced_rank"],
    )
    parser.add_argument("--region_width", type=int, default=8)
    parser.add_argument("--n_layers", type=int, default=32)
    parser.add_argument(
        "--output_root",
        type=Path,
        default=Path("outputs/parameter_matched_localization"),
    )
    parser.add_argument("--model_name", default=DEFAULT_MODEL)
    parser.add_argument("--python_exe", default=sys.executable)
    parser.add_argument("--train_script", default="src/train_fullstack_lora.py")
    parser.add_argument("--summarize_script", default="scripts/experiments/localization/summarize_localization_results.py")
    parser.add_argument("--target_modules", nargs="+", default=DEFAULT_TARGET_MODULES)
    parser.add_argument("--num_train_epochs", type=float, default=3)
    parser.add_argument("--learning_rate", default="2e-4")
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--torch_dtype", default="bfloat16", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip_summary", action="store_true")
    parser.add_argument(
        "--summary_only",
        action="store_true",
        help="Rebuild top-level summaries by discovering completed run subdirectories without training.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    if args.summary_only:
        specs = discover_completed_specs(
            output_root=args.output_root,
            controls=args.controls,
            n_layers=args.n_layers,
            region_width=args.region_width,
        )
        print(f"Discovered {len(specs)} completed parameter-matched runs under {args.output_root}")
        summarize_completed_runs(specs, args.output_root)
        return

    specs = build_specs(args)
    manifest_path = args.output_root / "parameter_matched_manifest.csv"
    write_manifest(specs, manifest_path)

    print(f"Wrote manifest: {manifest_path}")
    print(f"Planned runs: {len(specs)}")

    for spec in specs:
        print("\n" + "=" * 80)
        print(
            f"{spec.control} | {spec.objective} | budget={spec.budget} | "
            f"condition={spec.condition} | rank={spec.rank} | seed={spec.seed}"
        )

        if is_complete(spec) and not args.overwrite:
            print(f"[skip] Output appears complete: {spec.output_dir}")
            continue

        if not args.dry_run:
            spec.output_dir.mkdir(parents=True, exist_ok=True)

        run_command(spec.train_command, dry_run=args.dry_run)
        run_command(spec.eval_command, dry_run=args.dry_run)

        if not args.dry_run:
            enrich_result_jsonl(spec)

        run_command(spec.summarize_command, dry_run=args.dry_run)

        if not args.dry_run:
            (spec.output_dir / ".complete").write_text("complete\n", encoding="utf-8")

    if not args.dry_run and not args.skip_summary:
        summarize_completed_runs(specs, args.output_root)

    print("\nDone.")


if __name__ == "__main__":
    main()
