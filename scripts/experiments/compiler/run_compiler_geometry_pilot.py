#!/usr/bin/env python3
"""Dry-runnable episode x configuration x seed compiler-geometry pilot runner."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from compiler_common import (  # noqa: E402
    DEFAULT_TARGET_MODULES,
    OBJECTIVE_EVAL_SCRIPTS,
    model_slug,
    read_jsonl,
    shell_join,
    write_csv,
    write_jsonl,
)
from evaluate_preservation import validate_baseline_cache_file  # noqa: E402


@dataclass(frozen=True)
class PilotJob:
    episode_id: str
    config_id: str
    learning_type: str
    spec_id: str
    meta_split: str
    seed: int
    model_name: str
    model_slug: str
    output_dir: Path
    adapter_dir: Path
    result_jsonl: Path
    preservation_jsonl: Path | None
    preservation_summary_json: Path | None
    preservation_baseline_cache: Path | None
    summary_csv: Path
    run_manifest_path: Path
    metadata_path: Path
    complete_marker: Path
    train_command: List[str]
    eval_command: List[str]
    preservation_cache_command: List[str] | None
    preservation_command: List[str] | None
    summarize_command: List[str]

    @property
    def command(self) -> str:
        commands = [
            self.preservation_cache_command,
            self.train_command,
            self.eval_command,
            self.preservation_command,
            self.summarize_command,
        ]
        return " && ".join(shell_join(cmd) for cmd in commands if cmd)


def safe_episode_path(episode_id: str) -> str:
    return episode_id.replace("::", "__").replace("/", "_")


def write_one_episode_manifest(path: Path, episode: Dict[str, Any]) -> None:
    row = dict(episode)
    row["run_id"] = episode["episode_id"]
    write_jsonl(path, [row])


def select_episodes(
    episodes: Sequence[Dict[str, Any]],
    meta_split: str | None,
    learning_types: Sequence[str] | None,
    episode_ids: Sequence[str] | None,
    max_episodes_per_type: int | None,
) -> List[Dict[str, Any]]:
    selected = list(episodes)
    if meta_split:
        selected = [row for row in selected if row.get("meta_split") == meta_split]
    if learning_types:
        allowed = set(learning_types)
        selected = [row for row in selected if row.get("learning_type") in allowed]
    if episode_ids:
        allowed_ids = set(episode_ids)
        selected = [row for row in selected if row.get("episode_id") in allowed_ids]

    selected = sorted(selected, key=lambda row: (row["learning_type"], row["episode_id"]))
    if max_episodes_per_type is not None:
        counts: Dict[str, int] = {}
        limited: List[Dict[str, Any]] = []
        for row in selected:
            lt = row["learning_type"]
            counts[lt] = counts.get(lt, 0)
            if counts[lt] < max_episodes_per_type:
                limited.append(row)
                counts[lt] += 1
        selected = limited
    return selected


def select_configs(
    configs: Sequence[Dict[str, Any]],
    config_ids: Sequence[str] | None,
) -> List[Dict[str, Any]]:
    if config_ids:
        allowed = set(config_ids)
        configs = [row for row in configs if row["config_id"] in allowed]
    return sorted(configs, key=lambda row: row["config_id"])


def build_jobs(args: argparse.Namespace) -> List[PilotJob]:
    episodes = select_episodes(
        read_jsonl(args.episode_manifest),
        meta_split=args.meta_split,
        learning_types=args.learning_types,
        episode_ids=args.episode_ids,
        max_episodes_per_type=args.max_episodes_per_type,
    )
    configs = select_configs(read_jsonl(args.config_library), args.config_ids)
    if not episodes:
        raise ValueError("No episodes selected.")
    if not configs:
        raise ValueError("No configs selected.")

    slug = model_slug(args.model_name_or_path)
    preservation_baseline_cache = None
    if args.preservation_examples_path:
        preservation_baseline_cache = Path(
            args.preservation_baseline_cache
            or (args.results_root / f"preservation_baseline_{slug}.jsonl")
        )
    jobs: List[PilotJob] = []
    for episode in episodes:
        learning_type = episode["learning_type"]
        if learning_type not in OBJECTIVE_EVAL_SCRIPTS:
            raise ValueError(f"No evaluator configured for {learning_type}")
        spec_id = episode["spec_ids"][0]
        for config in configs:
            for seed in args.seeds:
                rel = (
                    Path(safe_episode_path(episode["episode_id"]))
                    / config["config_id"]
                    / f"seed_{seed}"
                )
                output_dir = args.results_root / rel
                adapter_dir = args.output_root / rel / "adapter"
                run_manifest_path = output_dir / "episode_run_manifest.jsonl"
                result_jsonl = output_dir / "evaluation.jsonl"
                preservation_jsonl = output_dir / "preservation.jsonl" if args.preservation_examples_path else None
                preservation_summary_json = output_dir / "preservation_summary.json" if args.preservation_examples_path else None
                summary_csv = output_dir / "summary_localization_by_seed.csv"
                metadata_path = output_dir / "compiler_job_metadata.json"
                complete_marker = output_dir / ".complete"

                train_cmd: List[str] = [
                    args.python_exe,
                    args.train_script,
                    "--model_name",
                    args.model_name_or_path,
                    "--examples_path",
                    args.examples_path,
                    "--manifest_path",
                    str(run_manifest_path),
                    "--run_id",
                    episode["episode_id"],
                    "--output_dir",
                    str(adapter_dir),
                    "--target_modules",
                    *[str(x) for x in config.get("target_modules", args.target_modules)],
                    "--lora_r",
                    str(config.get("lora_r", args.lora_r)),
                    "--lora_alpha",
                    str(config.get("lora_alpha", args.lora_alpha)),
                    "--lora_dropout",
                    str(config.get("lora_dropout", args.lora_dropout)),
                    "--num_train_epochs",
                    str(args.num_train_epochs),
                    "--save_strategy",
                    args.save_strategy,
                    "--learning_rate",
                    str(args.learning_rate),
                    "--per_device_train_batch_size",
                    str(args.per_device_train_batch_size),
                    "--gradient_accumulation_steps",
                    str(args.gradient_accumulation_steps),
                    "--max_length",
                    str(args.max_length),
                    "--seed",
                    str(seed),
                    "--torch_dtype",
                    args.torch_dtype,
                    "--device_map",
                    args.device_map,
                    "--localization_condition",
                    str(config["localization_condition"]),
                    "--region_width",
                    str(config.get("region_width", args.region_width)),
                ]
                if config.get("localization_condition") == "layers" and config.get("layer_indices"):
                    train_cmd.extend(
                        ["--layer_indices", ",".join(str(x) for x in config["layer_indices"])]
                    )
                if config.get("layers_pattern"):
                    train_cmd.extend(["--layers_pattern", str(config["layers_pattern"])])
                if args.bf16:
                    train_cmd.append("--bf16")
                if args.fp16:
                    train_cmd.append("--fp16")
                if args.gradient_checkpointing:
                    train_cmd.append("--gradient_checkpointing")

                eval_cmd = [
                    args.python_exe,
                    OBJECTIVE_EVAL_SCRIPTS[learning_type],
                    "--model_name",
                    args.model_name_or_path,
                    "--adapter_path",
                    str(adapter_dir),
                    "--examples_path",
                    args.examples_path,
                    "--manifest_path",
                    str(run_manifest_path),
                    "--run_id",
                    episode["episode_id"],
                    "--output_path",
                    str(result_jsonl),
                    "--max_new_tokens",
                    str(args.max_new_tokens),
                    "--device_map",
                    args.device_map,
                    "--torch_dtype",
                    args.torch_dtype,
                ]

                preservation_cache_cmd = None
                preservation_cmd = None
                if args.preservation_examples_path:
                    assert preservation_baseline_cache is not None
                    assert preservation_jsonl is not None
                    assert preservation_summary_json is not None
                    preservation_cache_cmd = [
                        args.python_exe,
                        args.preservation_script,
                        "--mode",
                        "cache_frozen",
                        "--model_name",
                        args.model_name_or_path,
                        "--preservation_examples_path",
                        args.preservation_examples_path,
                        "--baseline_cache",
                        str(preservation_baseline_cache),
                        "--max_new_tokens",
                        str(args.preservation_max_new_tokens),
                        "--prompt_format",
                        args.preservation_prompt_format,
                        "--response_extraction",
                        args.preservation_response_extraction,
                        "--device_map",
                        args.device_map,
                        "--torch_dtype",
                        args.torch_dtype,
                        "--empty_cuda_cache_every",
                        str(getattr(args, "preservation_empty_cuda_cache_every", 0)),
                    ]
                    if getattr(args, "preservation_disable_generation_cache", False):
                        preservation_cache_cmd.append("--disable_generation_cache")
                    preservation_cmd = [
                        args.python_exe,
                        args.preservation_script,
                        "--mode",
                        "evaluate",
                        "--model_name",
                        args.model_name_or_path,
                        "--adapter_path",
                        str(adapter_dir),
                        "--preservation_examples_path",
                        args.preservation_examples_path,
                        "--baseline_cache",
                        str(preservation_baseline_cache),
                        "--output_path",
                        str(preservation_jsonl),
                        "--summary_path",
                        str(preservation_summary_json),
                        "--episode_id",
                        episode["episode_id"],
                        "--config_id",
                        config["config_id"],
                        "--seed",
                        str(seed),
                        "--max_new_tokens",
                        str(args.preservation_max_new_tokens),
                        "--prompt_format",
                        args.preservation_prompt_format,
                        "--response_extraction",
                        args.preservation_response_extraction,
                        "--device_map",
                        args.device_map,
                        "--torch_dtype",
                        args.torch_dtype,
                        "--empty_cuda_cache_every",
                        str(getattr(args, "preservation_empty_cuda_cache_every", 0)),
                        "--forbidden_spec_ids",
                        *[str(x) for x in episode.get("spec_ids", [])],
                    ]
                    if getattr(args, "preservation_disable_generation_cache", False):
                        preservation_cmd.append("--disable_generation_cache")

                summarize_cmd = [
                    args.python_exe,
                    args.summarize_script,
                    "--inputs",
                    str(result_jsonl),
                    "--output_csv",
                    str(summary_csv),
                ]

                jobs.append(
                    PilotJob(
                        episode_id=episode["episode_id"],
                        config_id=config["config_id"],
                        learning_type=learning_type,
                        spec_id=spec_id,
                        meta_split=episode["meta_split"],
                        seed=int(seed),
                        model_name=args.model_name_or_path,
                        model_slug=slug,
                        output_dir=output_dir,
                        adapter_dir=adapter_dir,
                        result_jsonl=result_jsonl,
                        preservation_jsonl=preservation_jsonl,
                        preservation_summary_json=preservation_summary_json,
                        preservation_baseline_cache=preservation_baseline_cache,
                        summary_csv=summary_csv,
                        run_manifest_path=run_manifest_path,
                        metadata_path=metadata_path,
                        complete_marker=complete_marker,
                        train_command=[str(x) for x in train_cmd],
                        eval_command=[str(x) for x in eval_cmd],
                        preservation_cache_command=[str(x) for x in preservation_cache_cmd] if preservation_cache_cmd else None,
                        preservation_command=[str(x) for x in preservation_cmd] if preservation_cmd else None,
                        summarize_command=[str(x) for x in summarize_cmd],
                    )
                )
    return jobs


def metadata_for(job: PilotJob, episode: Dict[str, Any], config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "episode_id": job.episode_id,
        "spec_ids": episode["spec_ids"],
        "meta_split": episode["meta_split"],
        "learning_type": episode["learning_type"],
        "model_name": job.model_name,
        "model_slug": job.model_slug,
        "config_id": job.config_id,
        "localization_condition": config["localization_condition"],
        "selected_layers": config.get("resolved_layer_indices") if config.get("resolved_layer_indices") is not None else "all",
        "target_modules": config.get("target_modules", args.target_modules),
        "lora_r": config.get("lora_r", args.lora_r),
        "lora_alpha": config.get("lora_alpha", args.lora_alpha),
        "lora_dropout": config.get("lora_dropout", args.lora_dropout),
        "seed": job.seed,
        "train_budget_per_spec": episode["train_budget_per_spec"],
        "num_train_examples": len(episode["train_example_ids"]),
        "approximate_parameter_cost": config.get("approximate_parameter_cost"),
        "trainable_parameters": None,
        "training_hyperparameters": {
            "num_train_epochs": args.num_train_epochs,
            "save_strategy": args.save_strategy,
            "learning_rate": args.learning_rate,
            "per_device_train_batch_size": args.per_device_train_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "max_length": args.max_length,
            "torch_dtype": args.torch_dtype,
        },
        "git_commit": git_commit(),
        "preservation_examples_path": args.preservation_examples_path,
        "preservation_baseline_cache": str(job.preservation_baseline_cache) if job.preservation_baseline_cache else None,
        "preservation_summary_json": str(job.preservation_summary_json) if job.preservation_summary_json else None,
        "preservation_max_new_tokens": args.preservation_max_new_tokens,
        "preservation_empty_cuda_cache_every": getattr(args, "preservation_empty_cuda_cache_every", 0),
        "preservation_disable_generation_cache": getattr(args, "preservation_disable_generation_cache", False),
        "preservation_prompt_format": args.preservation_prompt_format,
        "preservation_response_extraction": args.preservation_response_extraction,
    }


def git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def write_manifest_csv(jobs: Sequence[PilotJob], path: Path) -> None:
    rows = [
        {
            "episode_id": job.episode_id,
            "config_id": job.config_id,
            "learning_type": job.learning_type,
            "spec_id": job.spec_id,
            "meta_split": job.meta_split,
            "seed": job.seed,
            "model_name": job.model_name,
            "model_slug": job.model_slug,
            "output_dir": str(job.output_dir),
            "adapter_dir": str(job.adapter_dir),
            "result_jsonl": str(job.result_jsonl),
            "preservation_jsonl": str(job.preservation_jsonl) if job.preservation_jsonl else None,
            "preservation_summary_json": str(job.preservation_summary_json) if job.preservation_summary_json else None,
            "preservation_baseline_cache": str(job.preservation_baseline_cache) if job.preservation_baseline_cache else None,
            "summary_csv": str(job.summary_csv),
            "run_manifest_path": str(job.run_manifest_path),
            "metadata_path": str(job.metadata_path),
            "command": job.command,
        }
        for job in jobs
    ]
    write_csv(
        path,
        rows,
        [
            "episode_id",
            "config_id",
            "learning_type",
            "spec_id",
            "meta_split",
            "seed",
            "model_name",
            "model_slug",
            "output_dir",
            "adapter_dir",
            "result_jsonl",
            "summary_csv",
            "preservation_jsonl",
            "preservation_summary_json",
            "preservation_baseline_cache",
            "run_manifest_path",
            "metadata_path",
            "command",
        ],
    )


def is_complete(job: PilotJob) -> bool:
    required = [job.complete_marker, job.result_jsonl, job.summary_csv]
    if job.preservation_jsonl is not None:
        required.append(job.preservation_jsonl)
    if job.preservation_summary_json is not None:
        required.append(job.preservation_summary_json)
    return all(path.exists() for path in required)


def run_command(
    cmd: Sequence[str],
    dry_run: bool,
    env_updates: Dict[str, str] | None = None,
) -> None:
    print(f"\n$ {shell_join(cmd)}", flush=True)
    if env_updates:
        print(
            "[env] "
            + " ".join(f"{key}={value}" for key, value in sorted(env_updates.items())),
            flush=True,
        )
    if not dry_run:
        env = os.environ.copy()
        if env_updates:
            env.update(env_updates)
        subprocess.run(list(cmd), check=True, env=env)


def can_reuse_existing_eval(job: PilotJob) -> bool:
    return job.adapter_dir.exists() and job.result_jsonl.exists()


def preservation_retry_env(args: argparse.Namespace) -> Dict[str, str]:
    env: Dict[str, str] = {}
    if getattr(args, "preservation_retry_cuda_launch_blocking", False):
        env["CUDA_LAUNCH_BLOCKING"] = "1"
    allocator_conf = getattr(args, "preservation_retry_allocator_conf", None)
    if allocator_conf:
        env["PYTORCH_CUDA_ALLOC_CONF"] = str(allocator_conf)
    return env


def run_preservation_command(job: PilotJob, args: argparse.Namespace) -> None:
    if not job.preservation_command:
        return
    try:
        run_command(job.preservation_command, args.dry_run)
    except subprocess.CalledProcessError:
        if not getattr(args, "retry_failed_preservation_once", False):
            raise
        retry_sleep = float(getattr(args, "preservation_retry_sleep_seconds", 0.0) or 0.0)
        if retry_sleep > 0:
            print(f"[retry] Preservation failed; waiting {retry_sleep:g}s before one retry.", flush=True)
            time.sleep(retry_sleep)
        env_updates = preservation_retry_env(args)
        print("[retry] Re-running preservation evaluation once with retry diagnostics.", flush=True)
        run_command(job.preservation_command, args.dry_run, env_updates=env_updates)


def enrich_results(job: PilotJob) -> None:
    if not job.result_jsonl.exists():
        return
    rows = []
    for row in read_jsonl(job.result_jsonl):
        out = dict(row)
        out["episode_id"] = job.episode_id
        out["config_id"] = job.config_id
        out["meta_split"] = job.meta_split
        out["compiler_seed"] = job.seed
        out["seed"] = job.seed
        out["model_name"] = job.model_name
        out["model_slug"] = job.model_slug
        rows.append(out)
    write_jsonl(job.result_jsonl, rows)


def run_jobs(jobs: Sequence[PilotJob], args: argparse.Namespace) -> None:
    episodes = {row["episode_id"]: row for row in read_jsonl(args.episode_manifest)}
    configs = {row["config_id"]: row for row in read_jsonl(args.config_library)}
    attempted_preservation_caches: set[Path] = set()
    for job in jobs:
        print("\n" + "=" * 80)
        print(f"{job.episode_id} | {job.config_id} | seed={job.seed}")
        if args.skip_existing and is_complete(job):
            print(f"[skip] Complete: {job.output_dir}")
            continue
        if job.preservation_cache_command and job.preservation_baseline_cache:
            cache_path = job.preservation_baseline_cache
            if cache_path not in attempted_preservation_caches:
                attempted_preservation_caches.add(cache_path)
                if cache_path.exists() and not args.dry_run:
                    validate_baseline_cache_file(
                        cache_path,
                        preservation_examples_path=args.preservation_examples_path,
                        model_name=job.model_name,
                    )
                    print(f"[skip] Preservation baseline cache exists and validates: {cache_path}")
                else:
                    run_command(job.preservation_cache_command, args.dry_run)
        job.output_dir.mkdir(parents=True, exist_ok=True)
        job.adapter_dir.parent.mkdir(parents=True, exist_ok=True)
        write_one_episode_manifest(job.run_manifest_path, episodes[job.episode_id])
        reuse_existing_eval = getattr(args, "reuse_existing_eval", False) and can_reuse_existing_eval(job)
        if reuse_existing_eval and job.metadata_path.exists():
            print(f"[resume] Keeping existing job metadata: {job.metadata_path}", flush=True)
        else:
            job.metadata_path.write_text(
                json.dumps(metadata_for(job, episodes[job.episode_id], configs[job.config_id], args), indent=2),
                encoding="utf-8",
            )
        if reuse_existing_eval:
            print(
                f"[resume] Reusing existing adapter and objective evaluation: {job.output_dir}",
                flush=True,
            )
        else:
            run_command(job.train_command, args.dry_run)
            run_command(job.eval_command, args.dry_run)
        if not args.dry_run:
            enrich_results(job)
        if job.preservation_command:
            run_preservation_command(job, args)
        run_command(job.summarize_command, args.dry_run)
        if not args.dry_run:
            job.complete_marker.write_text("complete\n", encoding="utf-8")
            if args.cleanup_adapter_after_eval and job.adapter_dir.exists():
                print(f"[cleanup] Removing adapter: {job.adapter_dir}", flush=True)
                shutil.rmtree(job.adapter_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run compiler geometry pilot jobs.")
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--episode_manifest", default="data/compiler/episode_manifest.jsonl")
    parser.add_argument("--config_library", default="data/compiler/config_library.jsonl")
    parser.add_argument("--examples_path", default="data/prompt_examples.jsonl")
    parser.add_argument("--meta_split", default=None, choices=["train", "validation", "test"])
    parser.add_argument("--learning_types", nargs="+", default=None)
    parser.add_argument("--episode_ids", nargs="+", default=None)
    parser.add_argument("--config_ids", nargs="+", default=None)
    parser.add_argument("--max_episodes_per_type", type=int, default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=[11])
    parser.add_argument("--output_root", type=Path, default=Path("outputs/compiler/pilot"))
    parser.add_argument("--results_root", type=Path, default=Path("outputs/compiler/pilot"))
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument(
        "--cleanup_adapter_after_eval",
        action="store_true",
        help="Delete each final adapter after successful evaluation, summarization, and completion marking.",
    )
    parser.add_argument(
        "--save_strategy",
        default="no",
        choices=["no", "steps", "epoch"],
        help="Transformers Trainer checkpoint save strategy. Compiler pilots default to 'no' to avoid intermediate checkpoints.",
    )
    parser.add_argument("--python_exe", default=sys.executable)
    parser.add_argument("--train_script", default="src/train_fullstack_lora.py")
    parser.add_argument("--summarize_script", default="scripts/experiments/localization/summarize_localization_results.py")
    parser.add_argument("--preservation_script", default="src/evaluate_preservation.py")
    parser.add_argument("--target_modules", nargs="+", default=DEFAULT_TARGET_MODULES)
    parser.add_argument("--region_width", type=int, default=8)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--num_train_epochs", type=float, default=3.0)
    parser.add_argument("--learning_rate", default="2e-4")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--torch_dtype", default="bfloat16", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--preservation_examples_path", default=None)
    parser.add_argument("--preservation_baseline_cache", default=None)
    parser.add_argument("--preservation_max_new_tokens", type=int, default=64)
    parser.add_argument(
        "--preservation_empty_cuda_cache_every",
        type=int,
        default=0,
        help="Forwarded to evaluate_preservation.py to lower preservation generation memory pressure.",
    )
    parser.add_argument(
        "--preservation_disable_generation_cache",
        action="store_true",
        help="Forwarded to evaluate_preservation.py; lowers preservation peak memory by disabling KV cache.",
    )
    parser.add_argument(
        "--retry_failed_preservation_once",
        action="store_true",
        help="Retry a failed preservation subprocess once before failing the job.",
    )
    parser.add_argument(
        "--preservation_retry_cuda_launch_blocking",
        action="store_true",
        help="Set CUDA_LAUNCH_BLOCKING=1 only for the preservation retry subprocess.",
    )
    parser.add_argument(
        "--preservation_retry_allocator_conf",
        default=None,
        help="Optional PYTORCH_CUDA_ALLOC_CONF value used only for the preservation retry subprocess.",
    )
    parser.add_argument(
        "--preservation_retry_sleep_seconds",
        type=float,
        default=5.0,
        help="Seconds to wait before the one-shot preservation retry.",
    )
    parser.add_argument(
        "--reuse_existing_eval",
        action="store_true",
        help="If adapter_dir and evaluation.jsonl already exist, resume at preservation/summarization.",
    )
    parser.add_argument("--preservation_prompt_format", choices=["auto", "project", "chat", "raw"], default="auto")
    parser.add_argument("--preservation_response_extraction", choices=["first_line", "full"], default="first_line")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.results_root.mkdir(parents=True, exist_ok=True)
    jobs = build_jobs(args)
    manifest_path = args.results_root / "compiler_geometry_pilot_manifest.csv"
    write_manifest_csv(jobs, manifest_path)
    print(f"Wrote manifest: {manifest_path}")
    print(f"Planned jobs: {len(jobs)}")
    run_jobs(jobs, args)
    print("\nDone.")


if __name__ == "__main__":
    main()
