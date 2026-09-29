#!/usr/bin/env python3
"""Command orchestrator for the Gemma adaptation-compiler replication.

This script does not duplicate the Llama compiler implementation. It builds
Gemma-specific protocol files and emits commands that call the existing runner,
feature extractor, geometry-preparation, predictor, selection, and comparison
scripts.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from compiler_backbones import (  # noqa: E402
    HIGH_CAPACITY_CONFIG_ID,
    PRIMARY_COMPILER_CONFIG_IDS,
    build_backbone_config_library,
    resolve_backbone_spec,
)
from compiler_common import shell_join, write_jsonl  # noqa: E402


DEFAULT_MODEL = "google/gemma-2-9b-it"
DEFAULT_RESULTS_ROOT = Path("outputs/compiler_gemma")
DEFAULT_OUTPUTS_ROOT = Path("outputs/compiler_gemma")
GA_CANDIDATES = [1, 2, 4, 8]


class ScheduleSelectionError(RuntimeError):
    """Raised when Gemma calibration did not produce an executable schedule."""


def command_count_for_phase(phase: str) -> int:
    if phase == "smoke":
        return 2
    if phase == "calibration":
        return 25 * len(GA_CANDIDATES)
    if phase == "pilot":
        return 100 * 4
    if phase == "full":
        return (400 * 4 * 1) + (100 * 4 * 1) + (100 * 4 * 3)
    return 0


def schedule_candidate_table(payload: Dict[str, Any]) -> str:
    rows = payload.get("candidate_summaries") or []
    if not rows:
        return "No candidate_summaries were recorded."
    lines = ["Calibration candidates:"]
    for row in rows:
        lines.append(
            "  GA={ga}: A={a:.3f} T={t:.3f} B={b:.3f} P={p:.3f} U={u:.3f} failures={f}".format(
                ga=int(row.get("grad_accum")),
                a=float(row.get("acquisition") or 0.0),
                t=float(row.get("transfer") or 0.0),
                b=float(row.get("boundedness") or 0.0),
                p=float(row.get("preservation") or 0.0),
                u=float(row.get("balanced_utility") or 0.0),
                f=int(row.get("failures") or 0),
            )
        )
    return "\n".join(lines)


def read_selected_grad_accum(
    path: Path,
    fallback: int,
    *,
    allow_fallback: bool,
    forced_grad_accum: int | None = None,
) -> int:
    if forced_grad_accum is not None:
        print(
            "[override] Using explicitly requested "
            f"gradient_accumulation_steps={forced_grad_accum}; calibration selection is bypassed."
        )
        return int(forced_grad_accum)
    if not path.exists():
        if allow_fallback:
            print(
                f"[warn] {path} does not exist; using uncalibrated "
                f"gradient_accumulation_steps={fallback}."
            )
            return int(fallback)
        raise ScheduleSelectionError(
            f"{path} does not exist. Run Gemma calibration first, or pass "
            "--force_gradient_accumulation_steps after choosing a schedule explicitly."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "selected":
        if allow_fallback:
            print(
                f"[warn] {path} status is {payload.get('status')!r}; using uncalibrated "
                f"gradient_accumulation_steps={fallback}."
            )
            return int(fallback)
        raise ScheduleSelectionError(
            f"{path} has status={payload.get('status')!r}; no schedule was auto-selected.\n"
            f"Reason: {payload.get('reason', 'not recorded')}\n"
            f"{schedule_candidate_table(payload)}\n"
            "Next step: inspect the calibration tradeoff, then either rerun calibration with explicit "
            "selection gates or pass --force_gradient_accumulation_steps <N> to pilot/full."
        )
    return int(payload["gradient_accumulation_steps"])


def grad_accum_for_phase(args: argparse.Namespace) -> int:
    if args.phase in {"smoke", "calibration"}:
        return int(args.gradient_accumulation_steps)
    return read_selected_grad_accum(
        Path(args.selected_schedule),
        args.gradient_accumulation_steps,
        allow_fallback=args.allow_uncalibrated_schedule,
        forced_grad_accum=args.force_gradient_accumulation_steps,
    )


def make_config_library(args: argparse.Namespace) -> None:
    spec = resolve_backbone_spec(
        args.model_name_or_path,
        n_layers=args.n_layers,
        target_modules=args.target_modules,
        layers_pattern=args.layers_pattern,
        torch_dtype=args.torch_dtype,
        device_map=args.device_map,
    )
    rows = build_backbone_config_library(spec)
    output = Path(args.config_library)
    write_jsonl(output, rows)
    print(f"Wrote {len(rows)} Gemma compiler configs to {output}")
    for row in rows:
        print(
            f"  {row['config_id']}: condition={row['localization_condition']} "
            f"rank={row['lora_r']} layers={row['n_layers_adapted']} "
            f"cost={row['approximate_parameter_cost']}"
        )


def runner_command(
    args: argparse.Namespace,
    *,
    results_root: Path,
    output_root: Path,
    meta_split: str,
    config_ids: Sequence[str],
    seeds: Sequence[int],
    gradient_accumulation_steps: int,
    max_episodes_per_type: int | None = None,
    learning_types: Sequence[str] | None = None,
    episode_ids: Sequence[str] | None = None,
    num_train_epochs: float | None = None,
) -> List[str]:
    spec = resolve_backbone_spec(
        args.model_name_or_path,
        n_layers=args.n_layers,
        target_modules=args.target_modules,
        layers_pattern=args.layers_pattern,
        torch_dtype=args.torch_dtype,
        device_map=args.device_map,
    )
    cmd = [
        args.python_exe,
        "scripts/experiments/compiler/run_compiler_geometry_pilot.py",
        "--model_name_or_path",
        args.model_name_or_path,
        "--episode_manifest",
        args.episode_manifest,
        "--config_library",
        args.config_library,
        "--examples_path",
        args.examples_path,
        "--meta_split",
        meta_split,
        "--config_ids",
        *config_ids,
        "--seeds",
        *[str(seed) for seed in seeds],
        "--output_root",
        str(output_root),
        "--results_root",
        str(results_root),
        "--num_train_epochs",
        str(num_train_epochs if num_train_epochs is not None else args.num_train_epochs),
        "--learning_rate",
        str(args.learning_rate),
        "--per_device_train_batch_size",
        str(args.per_device_train_batch_size),
        "--gradient_accumulation_steps",
        str(gradient_accumulation_steps),
        "--max_length",
        str(args.max_length),
        "--max_new_tokens",
        str(args.max_new_tokens),
        "--torch_dtype",
        args.torch_dtype,
        "--device_map",
        args.device_map,
        "--region_width",
        str(spec.region_width),
        "--target_modules",
        *list(spec.target_modules),
        "--preservation_examples_path",
        args.preservation_examples_path,
        "--preservation_baseline_cache",
        args.preservation_baseline_cache,
        "--preservation_max_new_tokens",
        str(args.preservation_max_new_tokens),
        "--preservation_empty_cuda_cache_every",
        str(args.preservation_empty_cuda_cache_every),
        "--save_strategy",
        "no",
        "--skip_existing",
        "--cleanup_adapter_after_eval",
    ]
    if args.preservation_disable_generation_cache:
        cmd.append("--preservation_disable_generation_cache")
    if args.retry_failed_preservation_once:
        cmd.append("--retry_failed_preservation_once")
    if args.preservation_retry_cuda_launch_blocking:
        cmd.append("--preservation_retry_cuda_launch_blocking")
    if args.preservation_retry_allocator_conf:
        cmd.extend(["--preservation_retry_allocator_conf", args.preservation_retry_allocator_conf])
    if args.preservation_retry_sleep_seconds is not None:
        cmd.extend(["--preservation_retry_sleep_seconds", str(args.preservation_retry_sleep_seconds)])
    if args.reuse_existing_eval:
        cmd.append("--reuse_existing_eval")
    if learning_types:
        cmd.extend(["--learning_types", *learning_types])
    if episode_ids:
        cmd.extend(["--episode_ids", *episode_ids])
    if max_episodes_per_type is not None:
        cmd.extend(["--max_episodes_per_type", str(max_episodes_per_type)])
    if args.gradient_checkpointing:
        cmd.append("--gradient_checkpointing")
    if args.runner_dry_run:
        cmd.append("--dry_run")
    return [str(part) for part in cmd]


def summarizer_command(args: argparse.Namespace, mode: str, *extra: str) -> List[str]:
    return [
        args.python_exe,
        "scripts/experiments/compiler/summarize_gemma_compiler_replication.py",
        mode,
        *extra,
    ]


def build_phase_commands(args: argparse.Namespace) -> List[List[str]]:
    results_root = Path(args.results_root)
    outputs_root = Path(args.outputs_root)

    if args.phase == "smoke":
        ga = grad_accum_for_phase(args)
        return [
            runner_command(
                args,
                results_root=results_root / "smoke" / "runs",
                output_root=outputs_root / "smoke",
                meta_split="train",
                learning_types=args.smoke_learning_types,
                max_episodes_per_type=args.smoke_episodes_per_type,
                config_ids=args.primary_config_ids[:2],
                seeds=[11],
                gradient_accumulation_steps=ga,
                num_train_epochs=args.smoke_num_train_epochs,
            ),
            summarizer_command(
                args,
                "prepare",
                "--input_glob",
                str(results_root / "smoke" / "runs" / "**" / "evaluation.jsonl"),
                "--output_dir",
                str(results_root / "smoke"),
                "--allow_incomplete_smoke_test",
            ),
        ]

    if args.phase == "calibration":
        commands: List[List[str]] = []
        for candidate in args.grad_accum_candidates:
            commands.append(
                runner_command(
                    args,
                    results_root=results_root / "calibration" / f"ga_{candidate}" / "runs",
                    output_root=outputs_root / "calibration" / f"ga_{candidate}",
                    meta_split="train",
                    max_episodes_per_type=5,
                    config_ids=[HIGH_CAPACITY_CONFIG_ID],
                    seeds=[11],
                    gradient_accumulation_steps=int(candidate),
                )
            )
        commands.append(
            summarizer_command(
                args,
                "calibration",
                "--results_root",
                str(results_root / "calibration"),
                "--output_dir",
                str(results_root / "calibration"),
                "--min_acquisition",
                str(args.calibration_min_acquisition),
                "--min_transfer",
                str(args.calibration_min_transfer),
                "--min_boundedness",
                str(args.calibration_min_boundedness),
                "--min_preservation",
                str(args.calibration_min_preservation),
                "--max_acquisition_drop",
                str(args.calibration_max_acquisition_drop),
                "--max_transfer_drop",
                str(args.calibration_max_transfer_drop),
                "--max_failures",
                str(args.calibration_max_failures),
            )
        )
        return commands

    if args.phase == "pilot":
        ga = grad_accum_for_phase(args)
        return [
            runner_command(
                args,
                results_root=results_root / "pilot" / "runs",
                output_root=outputs_root / "pilot",
                meta_split="train",
                max_episodes_per_type=20,
                config_ids=args.primary_config_ids,
                seeds=[11],
                gradient_accumulation_steps=ga,
            ),
            summarizer_command(
                args,
                "pilot",
                "--input_glob",
                str(results_root / "pilot" / "runs" / "**" / "evaluation.jsonl"),
                "--output_dir",
                str(results_root / "pilot"),
                "--config_ids",
                *args.primary_config_ids,
            ),
        ]

    if args.phase == "full":
        ga = grad_accum_for_phase(args)
        return [
            runner_command(
                args,
                results_root=results_root / "full" / "runs" / "train",
                output_root=outputs_root / "full" / "train",
                meta_split="train",
                max_episodes_per_type=80,
                config_ids=args.primary_config_ids,
                seeds=[11],
                gradient_accumulation_steps=ga,
            ),
            runner_command(
                args,
                results_root=results_root / "full" / "runs" / "validation",
                output_root=outputs_root / "full" / "validation",
                meta_split="validation",
                max_episodes_per_type=20,
                config_ids=args.primary_config_ids,
                seeds=[11],
                gradient_accumulation_steps=ga,
            ),
            runner_command(
                args,
                results_root=results_root / "full" / "runs" / "test",
                output_root=outputs_root / "full" / "test",
                meta_split="test",
                max_episodes_per_type=20,
                config_ids=args.primary_config_ids,
                seeds=[11, 22, 33],
                gradient_accumulation_steps=ga,
            ),
            summarizer_command(
                args,
                "prepare",
                "--input_glob",
                str(results_root / "full" / "runs" / "**" / "evaluation.jsonl"),
                "--output_dir",
                str(results_root / "full"),
                "--config_ids",
                *args.primary_config_ids,
            ),
            [
                args.python_exe,
                "scripts/reproduce/analyze_compiler_headroom.py",
                "--records",
                str(results_root / "full" / "raw_geometry.csv"),
                "--config_ids",
                *args.primary_config_ids,
                "--output_dir",
                str(results_root / "full" / "headroom_analysis"),
                "--eval_splits",
                "validation",
                "test",
            ],
        ]

    if args.phase == "features":
        return [
            [
                args.python_exe,
                "scripts/experiments/compiler/extract_episode_model_features.py",
                "--model_name_or_path",
                args.model_name_or_path,
                "--episode_manifest",
                args.episode_manifest,
                "--examples_path",
                args.examples_path,
                "--output_root",
                str(results_root / "full" / "features" / "model"),
                "--target_modules",
                *args.target_modules,
                "--n_layers",
                str(resolve_backbone_spec(args.model_name_or_path, n_layers=args.n_layers).n_layers),
                "--backend",
                args.feature_backend,
                "--max_length",
                str(args.max_length),
                "--torch_dtype",
                args.torch_dtype,
                "--device_map",
                args.device_map,
                "--skip_existing",
            ]
        ]

    if args.phase == "predict":
        return [
            [
                args.python_exe,
                "scripts/experiments/compiler/train_geometry_predictor.py",
                "--geometry",
                str(results_root / "full" / "geometry_dataset.csv"),
                "--features",
                str(results_root / "full" / "features" / "model"),
                "--config_library",
                args.config_library,
                "--feature_set",
                args.feature_set,
                "--model_type",
                "auto",
                "--output_dir",
                str(results_root / "full" / "experiment2" / "full_auto"),
                "--config_ids",
                *args.primary_config_ids,
            ]
        ]

    if args.phase == "select":
        return [
            [
                args.python_exe,
                "src/evaluate_compiler.py",
                "--input",
                str(results_root / "full" / "experiment2" / "full_auto" / "predictions_test.csv"),
                "--train_records",
                str(results_root / "full" / "geometry_dataset.csv"),
                "--output_csv",
                str(results_root / "full" / "experiment3" / "compiler_evaluation.csv"),
                "--output_json",
                str(results_root / "full" / "experiment3" / "compiler_evaluation_summary.json"),
            ]
        ]

    if args.phase == "compare":
        return [
            summarizer_command(
                args,
                "compare",
                "--gemma_root",
                str(results_root / "full"),
                "--llama_root",
                "artifacts/llama",
                "--output_dir",
                str(results_root / "comparison"),
            )
        ]

    raise ValueError(f"Unhandled phase: {args.phase}")


def print_or_execute(commands: Sequence[Sequence[str]], args: argparse.Namespace) -> None:
    print(f"Phase: {args.phase}")
    expected_runs = command_count_for_phase(args.phase)
    if expected_runs:
        print(f"Expected adaptation runs: {expected_runs}")
    print(f"Commands: {len(commands)}")
    for idx, cmd in enumerate(commands, start=1):
        print(f"\n[{idx}/{len(commands)}] $ {shell_join(cmd)}")
        if args.execute:
            subprocess.run(list(cmd), check=True)
    if not args.execute:
        print("\n[print-only] Re-run with --execute to launch these commands.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gemma adaptation-compiler replication command orchestrator.")
    parser.add_argument(
        "--phase",
        choices=["init", "smoke", "calibration", "pilot", "full", "features", "predict", "select", "compare"],
        required=True,
    )
    parser.add_argument("--execute", action="store_true", help="Actually run generated commands. Default only prints.")
    parser.add_argument(
        "--runner_dry_run",
        action="store_true",
        help="Pass --dry_run to scripts/experiments/compiler/run_compiler_geometry_pilot.py when executing/printing adaptation phases.",
    )
    parser.add_argument("--python_exe", default=sys.executable)
    parser.add_argument("--model_name_or_path", default=DEFAULT_MODEL)
    parser.add_argument("--n_layers", type=int, default=42)
    parser.add_argument("--episode_manifest", default="data/compiler/episode_manifest.jsonl")
    parser.add_argument("--examples_path", default="data/prompt_examples.jsonl")
    parser.add_argument("--config_library", default="data/compiler_gemma/config_library.jsonl")
    parser.add_argument("--results_root", default=str(DEFAULT_RESULTS_ROOT))
    parser.add_argument("--outputs_root", default=str(DEFAULT_OUTPUTS_ROOT))
    parser.add_argument("--selected_schedule", default="artifacts/gemma/calibration/selected_schedule.json")
    parser.add_argument("--preservation_examples_path", default="data/compiler/preservation_examples.jsonl")
    parser.add_argument("--preservation_baseline_cache", default="outputs/compiler_gemma/preservation/gemma_2_9b_it_baseline.jsonl")
    parser.add_argument("--primary_config_ids", nargs="+", default=PRIMARY_COMPILER_CONFIG_IDS)
    parser.add_argument("--target_modules", nargs="+", default=None)
    parser.add_argument("--layers_pattern", default="layers")
    parser.add_argument("--num_train_epochs", type=float, default=3.0)
    parser.add_argument("--smoke_num_train_epochs", type=float, default=0.25)
    parser.add_argument("--learning_rate", default="2e-4")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument(
        "--allow_uncalibrated_schedule",
        action="store_true",
        help="Allow pilot/full command planning with --gradient_accumulation_steps before selected_schedule.json exists.",
    )
    parser.add_argument(
        "--force_gradient_accumulation_steps",
        type=int,
        default=None,
        help=(
            "Explicitly use this GA value for pilot/full despite a missing or inconclusive "
            "selected_schedule.json. Use only after inspecting calibration_summary.csv."
        ),
    )
    parser.add_argument("--grad_accum_candidates", nargs="+", type=int, default=GA_CANDIDATES)
    parser.add_argument("--calibration_min_acquisition", type=float, default=0.5)
    parser.add_argument("--calibration_min_transfer", type=float, default=0.5)
    parser.add_argument("--calibration_min_boundedness", type=float, default=0.8)
    parser.add_argument("--calibration_min_preservation", type=float, default=0.8)
    parser.add_argument("--calibration_max_acquisition_drop", type=float, default=0.05)
    parser.add_argument("--calibration_max_transfer_drop", type=float, default=0.05)
    parser.add_argument("--calibration_max_failures", type=int, default=0)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--preservation_max_new_tokens", type=int, default=16)
    parser.add_argument(
        "--preservation_empty_cuda_cache_every",
        type=int,
        default=1,
        help="Forwarded to preservation evaluation; Gemma defaults to clearing CUDA cache after every example.",
    )
    parser.add_argument(
        "--preservation_disable_generation_cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Forwarded to preservation evaluation; enabled by default for Gemma memory stability.",
    )
    parser.add_argument(
        "--retry_failed_preservation_once",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Retry a failed preservation subprocess once before failing the phase.",
    )
    parser.add_argument(
        "--preservation_retry_cuda_launch_blocking",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Set CUDA_LAUNCH_BLOCKING=1 for the one-shot preservation retry.",
    )
    parser.add_argument(
        "--preservation_retry_allocator_conf",
        default="expandable_segments:True",
        help="PYTORCH_CUDA_ALLOC_CONF value used only for the preservation retry subprocess.",
    )
    parser.add_argument(
        "--preservation_retry_sleep_seconds",
        type=float,
        default=5.0,
        help="Seconds to wait before the one-shot preservation retry.",
    )
    parser.add_argument(
        "--reuse_existing_eval",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume interrupted Gemma jobs at preservation/summarization when adapter and evaluation already exist.",
    )
    parser.add_argument("--torch_dtype", default="bfloat16", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--feature_backend", choices=["frozen_model", "proxy"], default="frozen_model")
    parser.add_argument("--feature_set", choices=["episode", "frozen_behavior", "module_probes", "full"], default="full")
    parser.add_argument("--smoke_learning_types", nargs="+", default=["lexical_binding"])
    parser.add_argument("--smoke_episodes_per_type", type=int, default=1)
    args = parser.parse_args()
    if args.target_modules is None:
        args.target_modules = list(resolve_backbone_spec(args.model_name_or_path, n_layers=args.n_layers).target_modules)
    return args


def main() -> None:
    args = parse_args()
    try:
        if args.phase == "init":
            make_config_library(args)
            return
        commands = build_phase_commands(args)
        print_or_execute(commands, args)
    except ScheduleSelectionError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
