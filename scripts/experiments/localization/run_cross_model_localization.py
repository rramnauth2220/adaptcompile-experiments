#!/usr/bin/env python3
"""Run cross-model localized adaptation robustness experiments."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, Iterable, List, Sequence


DEFAULT_MODELS = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "google/gemma-2-9b-it",
    "allenai/OLMo-2-1124-7B-Instruct",
    "Qwen/Qwen2.5-14B-Instruct",
]

MODEL_LAYER_COUNTS = {
    "meta-llama/Llama-3.1-8B-Instruct": 32,
    "mistralai/Mistral-7B-Instruct-v0.3": 32,
    "google/gemma-2-9b-it": 42,
    "allenai/OLMo-2-1124-7B-Instruct": 32,
    "Qwen/Qwen2.5-14B-Instruct": 48,
}

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
# These defaults cover Llama-style module names and work for many recent
# decoder-only families, but model families can diverge here. Override
# --target_modules when a family uses different projection/MLP names. The
# evaluator/trainer should also be checked for chat-template behavior when
# adding a new instruction-tuned model.
DEFAULT_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


@dataclass(frozen=True)
class RunSpec:
    model_name: str
    model_slug: str
    objective: str
    budget: int
    condition: str
    seed: int
    rank: int
    output_dir: Path
    adapter_dir: Path
    result_jsonl: Path
    per_run_summary_csv: Path
    run_id: str
    calibration_run_id: str
    train_command: List[str]
    eval_command: List[str]
    summarize_command: List[str]

    @property
    def manifest_command(self) -> str:
        return " && ".join(
            [shell_join(self.train_command), shell_join(self.eval_command), shell_join(self.summarize_command)]
        )


def model_slug(model_name: str) -> str:
    slug = model_name.split("/")[-1].lower()
    slug = slug.replace(".", "_")
    slug = re.sub(r"[^a-z0-9]+", "_", slug).strip("_")
    return slug


def shell_join(cmd: Sequence[Any]) -> str:
    parts = [str(x) for x in cmd]
    return subprocess.list2cmdline(parts) if os.name == "nt" else shlex.join(parts)


def run_command(cmd: Sequence[Any], dry_run: bool) -> None:
    print(f"\n$ {shell_join(cmd)}", flush=True)
    if not dry_run:
        subprocess.run([str(x) for x in cmd], check=True)


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, mode="rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, mode="wt", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_csv(path: str | Path) -> List[Dict[str, str]]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: str | Path, rows: List[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def maybe_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def sd(values: List[float]) -> float | None:
    return stdev(values) if len(values) > 1 else None


def parse_model_layers(values: Sequence[str] | None) -> Dict[str, int]:
    overrides: Dict[str, int] = {}
    for value in values or []:
        if "=" not in value:
            raise ValueError(f"--model_layers entries must look like MODEL=N, got {value!r}")
        model, count = value.rsplit("=", 1)
        overrides[model] = int(count)
    return overrides


def layer_count_for(model_name: str, overrides: Dict[str, int]) -> int:
    if model_name in overrides:
        return overrides[model_name]
    if model_name in MODEL_LAYER_COUNTS:
        return MODEL_LAYER_COUNTS[model_name]
    raise ValueError(
        f"No layer count known for {model_name!r}. Pass --model_layers {model_name}=N "
        "so early/middle/late can use a normalized 25% depth window."
    )


def normalized_region_width(model_name: str, overrides: Dict[str, int]) -> int:
    return max(1, math.ceil(layer_count_for(model_name, overrides) * 0.25))


def is_complete(spec: RunSpec) -> bool:
    return (spec.output_dir / ".complete").exists() and spec.result_jsonl.exists() and spec.per_run_summary_csv.exists()


def enrich_result_jsonl(spec: RunSpec) -> None:
    rows = []
    for row in read_jsonl(spec.result_jsonl):
        out = dict(row)
        out["cross_model_run_id"] = spec.run_id
        out["model_name"] = spec.model_name
        out["model_slug"] = spec.model_slug
        out["localization_condition"] = spec.condition
        out["condition"] = spec.condition
        out["seed"] = spec.seed
        out["localization_seed"] = spec.seed
        out["rank"] = spec.rank
        out["lora_r"] = spec.rank
        out["adapter_path"] = str(spec.adapter_dir)
        rows.append(out)
    write_jsonl(spec.result_jsonl, rows)


def build_specs(args: argparse.Namespace) -> List[RunSpec]:
    overrides = parse_model_layers(args.model_layers)
    specs: List[RunSpec] = []
    for model_name in args.models:
        slug = model_slug(model_name)
        region_width = normalized_region_width(model_name, overrides)
        for objective in args.objectives:
            budget = OBJECTIVE_BUDGETS[objective]
            calibration_run_id = f"calib::{objective}::specs25::budget{budget}"
            eval_script = OBJECTIVE_EVAL_SCRIPTS[objective]
            for condition in args.conditions:
                for seed in args.seeds:
                    rank = args.base_rank
                    run_id = f"crossmodel::{slug}::{objective}::budget{budget}::{condition}::rank{rank}::seed{seed}"
                    safe = f"{slug}_{objective}_budget{budget}_{condition}_rank{rank}_seed{seed}"
                    output_dir = (
                        args.output_root
                        / slug
                        / objective
                        / f"budget_{budget}"
                        / f"condition_{condition}"
                        / f"rank_{rank}"
                        / f"seed_{seed}"
                    )
                    adapter_dir = output_dir / "adapter"
                    result_jsonl = output_dir / f"adapter_{safe}.jsonl"
                    per_run_summary_csv = output_dir / "summary_localization_by_seed.csv"

                    train_cmd: List[Any] = [
                        args.python_exe,
                        args.train_script,
                        "--model_name",
                        model_name,
                        "--run_id",
                        calibration_run_id,
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
                        region_width,
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
                        model_name,
                        "--adapter_path",
                        adapter_dir,
                        "--run_id",
                        calibration_run_id,
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
                            model_name=model_name,
                            model_slug=slug,
                            objective=objective,
                            budget=budget,
                            condition=condition,
                            seed=seed,
                            rank=rank,
                            output_dir=output_dir,
                            adapter_dir=adapter_dir,
                            result_jsonl=result_jsonl,
                            per_run_summary_csv=per_run_summary_csv,
                            run_id=run_id,
                            calibration_run_id=calibration_run_id,
                            train_command=[str(x) for x in train_cmd],
                            eval_command=[str(x) for x in eval_cmd],
                            summarize_command=[str(x) for x in summarize_cmd],
                        )
                    )
    return specs


def write_manifest(specs: Sequence[RunSpec], path: Path) -> None:
    rows = [
        {
            "model_name": spec.model_name,
            "model_slug": spec.model_slug,
            "objective": spec.objective,
            "budget": spec.budget,
            "condition": spec.condition,
            "seed": spec.seed,
            "rank": spec.rank,
            "output_dir": str(spec.output_dir),
            "adapter_dir": str(spec.adapter_dir),
            "result_jsonl": str(spec.result_jsonl),
            "per_run_summary_csv": str(spec.per_run_summary_csv),
            "run_id": spec.run_id,
            "command": spec.manifest_command,
        }
        for spec in specs
    ]
    write_csv(
        path,
        rows,
        [
            "model_name",
            "model_slug",
            "objective",
            "budget",
            "condition",
            "seed",
            "rank",
            "output_dir",
            "adapter_dir",
            "result_jsonl",
            "per_run_summary_csv",
            "run_id",
            "command",
        ],
    )


def model_name_for_slug(slug: str) -> str:
    for model_name in DEFAULT_MODELS:
        if model_slug(model_name) == slug:
            return model_name
    return slug


def specs_from_manifest(path: Path) -> List[RunSpec]:
    specs: List[RunSpec] = []
    if not path.exists():
        return specs
    for row in read_csv(path):
        output_dir = Path(row["output_dir"])
        adapter_dir = Path(row["adapter_dir"])
        result_jsonl = Path(row["result_jsonl"])
        per_run_summary_csv = Path(row["per_run_summary_csv"])
        specs.append(
            RunSpec(
                model_name=row["model_name"],
                model_slug=row["model_slug"],
                objective=row["objective"],
                budget=int(row["budget"]),
                condition=row["condition"],
                seed=int(row["seed"]),
                rank=int(row["rank"]),
                output_dir=output_dir,
                adapter_dir=adapter_dir,
                result_jsonl=result_jsonl,
                per_run_summary_csv=per_run_summary_csv,
                run_id=row["run_id"],
                calibration_run_id="",
                train_command=[],
                eval_command=[],
                summarize_command=[],
            )
        )
    return specs


def parse_prefixed_int(value: str, prefix: str) -> int:
    if not value.startswith(prefix):
        raise ValueError(f"Expected {value!r} to start with {prefix!r}")
    return int(value[len(prefix) :])


def discover_specs(output_root: Path) -> List[RunSpec]:
    """Recover run specs from the on-disk cross-model result layout.

    This is deliberately independent of cross_model_manifest.csv because users
    often run one model/objective at a time, which overwrites the manifest for
    the most recent invocation. The directory layout is the durable record.
    """
    specs: List[RunSpec] = []
    for output_dir in sorted(output_root.glob("*/*/budget_*/condition_*/rank_*/seed_*")):
        if not output_dir.is_dir():
            continue
        try:
            rel = output_dir.relative_to(output_root)
            model_slug_value = rel.parts[0]
            objective = rel.parts[1]
            budget = parse_prefixed_int(rel.parts[2], "budget_")
            condition = rel.parts[3].removeprefix("condition_")
            rank = parse_prefixed_int(rel.parts[4], "rank_")
            seed = parse_prefixed_int(rel.parts[5], "seed_")
        except (IndexError, ValueError):
            continue
        if objective not in OBJECTIVE_BUDGETS or condition not in CONDITIONS:
            continue
        result_candidates = sorted(output_dir.glob("adapter_*.jsonl")) + sorted(output_dir.glob("adapter_*.jsonl.gz"))
        result_jsonl = result_candidates[0] if result_candidates else output_dir / "adapter_missing.jsonl"
        model_name = model_name_for_slug(model_slug_value)
        run_id = f"crossmodel::{model_slug_value}::{objective}::budget{budget}::{condition}::rank{rank}::seed{seed}"
        specs.append(
            RunSpec(
                model_name=model_name,
                model_slug=model_slug_value,
                objective=objective,
                budget=budget,
                condition=condition,
                seed=seed,
                rank=rank,
                output_dir=output_dir,
                adapter_dir=output_dir / "adapter",
                result_jsonl=result_jsonl,
                per_run_summary_csv=output_dir / "summary_localization_by_seed.csv",
                run_id=run_id,
                calibration_run_id="",
                train_command=[],
                eval_command=[],
                summarize_command=[],
            )
        )
    return specs


def merge_specs(specs: Sequence[RunSpec]) -> List[RunSpec]:
    by_key: Dict[tuple, RunSpec] = {}
    for spec in specs:
        key = (spec.model_slug, spec.objective, spec.budget, spec.condition, spec.rank, spec.seed)
        current = by_key.get(key)
        if current is None:
            by_key[key] = spec
            continue
        # Prefer the spec whose output files actually exist; this lets
        # filesystem discovery repair stale or overwritten manifest paths.
        current_score = int(current.per_run_summary_csv.exists()) + int(current.result_jsonl.exists())
        new_score = int(spec.per_run_summary_csv.exists()) + int(spec.result_jsonl.exists())
        if new_score >= current_score:
            by_key[key] = spec
    return [by_key[key] for key in sorted(by_key)]


def split_metric(rows: List[Dict[str, str]], split: str, metric: str) -> float | None:
    for row in rows:
        if row.get("split") == split and row.get("example_mode") == "ALL_MODES" and row.get("scoring_type") == "ALL_SCORING_TYPES":
            return maybe_float(row.get(metric))
    return None


def summarize_result_jsonl(spec: RunSpec) -> Dict[str, float] | None:
    if not spec.result_jsonl.exists():
        return None
    sums: Dict[tuple, Dict[str, float]] = {}
    counts: Dict[tuple, int] = {}
    for row in read_jsonl(spec.result_jsonl):
        split = str(row.get("split", "unknown"))
        key = (split, "ALL_MODES", "ALL_SCORING_TYPES")
        counts[key] = counts.get(key, 0) + 1
        bucket = sums.setdefault(key, {})
        for out_key, sources in [
            ("strict_accuracy", ["strict_accuracy", "strict_score", "strict_correct"]),
            ("concept_accuracy", ["concept_accuracy", "concept_score", "concept_correct"]),
            ("passed_accuracy", ["passed_accuracy", "passed", "passed_correct"]),
        ]:
            for source in sources:
                if source in row:
                    value = float(row[source]) if source.endswith("_accuracy") else float(bool(row[source]))
                    bucket[out_key] = bucket.get(out_key, 0.0) + value
                    break

    def mean_for(split: str, metric: str) -> float | None:
        key = (split, "ALL_MODES", "ALL_SCORING_TYPES")
        n = counts.get(key, 0)
        if n == 0 or metric not in sums.get(key, {}):
            return None
        return sums[key][metric] / n

    id_eval = mean_for("id_eval", "strict_accuracy")
    paraphrase = mean_for("paraphrase_eval", "strict_accuracy")
    transfer = mean_for("generalization", "strict_accuracy")
    boundedness = mean_for("negative_control", BOUND_SOURCE[spec.objective])
    if None in (id_eval, paraphrase, transfer, boundedness):
        return None
    return {
        "id_eval": id_eval,
        "paraphrase_eval": paraphrase,
        "transfer": transfer,
        "boundedness": boundedness,
    }


def extract_metrics(spec: RunSpec) -> Dict[str, float] | None:
    if spec.per_run_summary_csv.exists():
        rows = read_csv(spec.per_run_summary_csv)
        id_eval = split_metric(rows, "id_eval", "strict_accuracy")
        paraphrase = split_metric(rows, "paraphrase_eval", "strict_accuracy")
        transfer = split_metric(rows, "generalization", "strict_accuracy")
        boundedness = split_metric(rows, "negative_control", BOUND_SOURCE[spec.objective])
        if None not in (id_eval, paraphrase, transfer, boundedness):
            return {
                "id_eval": id_eval,
                "paraphrase_eval": paraphrase,
                "transfer": transfer,
                "boundedness": boundedness,
            }
    return summarize_result_jsonl(spec)


def summarize_completed_runs(specs: Sequence[RunSpec], output_root: Path) -> None:
    by_seed_rows: List[Dict[str, Any]] = []
    for spec in specs:
        if not spec.per_run_summary_csv.exists() and not spec.result_jsonl.exists():
            continue
        metrics = extract_metrics(spec)
        if metrics is None:
            print(f"[warn] Could not extract all metrics from {spec.per_run_summary_csv}")
            continue
        id_eval = metrics["id_eval"]
        paraphrase = metrics["paraphrase_eval"]
        transfer = metrics["transfer"]
        boundedness = metrics["boundedness"]
        by_seed_rows.append(
            {
                "model_name": spec.model_name,
                "model_slug": spec.model_slug,
                "objective": spec.objective,
                "budget": spec.budget,
                "condition": spec.condition,
                "seed": spec.seed,
                "rank": spec.rank,
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

    by_seed_path = output_root / "summary_cross_model_by_seed.csv"
    by_seed_fields = [
        "model_name",
        "model_slug",
        "objective",
        "budget",
        "condition",
        "seed",
        "rank",
        "id_eval",
        "paraphrase_eval",
        "acquisition",
        "transfer",
        "generalization",
        "boundedness",
        "output_dir",
        "summary_file",
    ]
    write_csv(by_seed_path, by_seed_rows, by_seed_fields)

    grouped: Dict[tuple, List[Dict[str, Any]]] = {}
    for row in by_seed_rows:
        key = (row["model_name"], row["model_slug"], row["objective"], row["budget"], row["condition"], row["rank"])
        grouped.setdefault(key, []).append(row)

    aggregate_rows: List[Dict[str, Any]] = []
    for key, rows in sorted(grouped.items()):
        model_name, slug, objective, budget, condition, rank = key

        def vals(metric: str) -> List[float]:
            return [float(row[metric]) for row in rows if row[metric] not in (None, "")]

        acquisition = vals("acquisition")
        transfer = vals("transfer")
        boundedness = vals("boundedness")
        aggregate_rows.append(
            {
                "model_name": model_name,
                "model_slug": slug,
                "objective": objective,
                "budget": budget,
                "condition": condition,
                "rank": rank,
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

    aggregate_path = output_root / "summary_cross_model_aggregate.csv"
    write_csv(
        aggregate_path,
        aggregate_rows,
        [
            "model_name",
            "model_slug",
            "objective",
            "budget",
            "condition",
            "rank",
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
    parser = argparse.ArgumentParser(description="Run cross-model localization robustness experiments.")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--model_layers", nargs="*", default=[], help="Optional MODEL=N overrides for normalized 25% windows.")
    parser.add_argument("--objectives", nargs="+", default=list(OBJECTIVE_BUDGETS), choices=list(OBJECTIVE_BUDGETS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 22, 33])
    parser.add_argument("--conditions", nargs="+", default=CONDITIONS, choices=CONDITIONS)
    parser.add_argument("--base_rank", type=int, default=8)
    parser.add_argument("--output_root", type=Path, default=Path("outputs/cross_model_localization"))
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip_summary", action="store_true")
    parser.add_argument("--summary_only", action="store_true", help="Rebuild top-level cross-model summaries without running training/eval.")
    parser.add_argument("--manifest_path", type=Path, default=None, help="Manifest to use with --summary_only. Defaults to OUTPUT_ROOT/cross_model_manifest.csv.")
    parser.add_argument("--no_discover_results", action="store_true", help="With --summary_only, use only the manifest instead of discovering result subdirectories.")
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    if args.summary_only:
        manifest_path = args.manifest_path or (args.output_root / "cross_model_manifest.csv")
        specs = specs_from_manifest(manifest_path)
        manifest_count = len(specs)
        discovered_count = 0
        if not args.no_discover_results:
            discovered = discover_specs(args.output_root)
            discovered_count = len(discovered)
            specs = merge_specs([*specs, *discovered])
        print(f"Loaded {manifest_count} manifest specs; discovered {discovered_count} result dirs; summarizing {len(specs)} unique runs.")
        summarize_completed_runs(specs, args.output_root)
        print("\nDone.")
        return
    specs = build_specs(args)
    manifest_path = args.output_root / "cross_model_manifest.csv"
    write_manifest(specs, manifest_path)
    print(f"Wrote manifest: {manifest_path}")
    print(f"Planned runs: {len(specs)}")

    for spec in specs:
        print("\n" + "=" * 80)
        print(f"{spec.model_slug} | {spec.objective} | budget={spec.budget} | {spec.condition} | rank={spec.rank} | seed={spec.seed}")
        if is_complete(spec) and not args.overwrite:
            print(f"[skip] Output appears complete: {spec.output_dir}")
            continue
        if not args.dry_run:
            spec.output_dir.mkdir(parents=True, exist_ok=True)
        run_command(spec.train_command, args.dry_run)
        run_command(spec.eval_command, args.dry_run)
        if not args.dry_run:
            enrich_result_jsonl(spec)
        run_command(spec.summarize_command, args.dry_run)
        if not args.dry_run:
            (spec.output_dir / ".complete").write_text("complete\n", encoding="utf-8")

    if not args.dry_run and not args.skip_summary:
        summarize_completed_runs(specs, args.output_root)
    print("\nDone.")


if __name__ == "__main__":
    main()
