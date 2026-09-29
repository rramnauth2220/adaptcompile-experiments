#!/usr/bin/env python3
"""
Train a LoRA adapter for one calibration-manifest row.

This is the original full-stack trainer with one added capability:
restrict LoRA updates to a selected layer region.

When --localization_condition full, behavior is equivalent to the original
full-stack calibration trainer, except for the added metadata and seed handling.

Examples:

Full stack:
  python src/train_fullstack_lora.py \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --run_id 'calib::lexical_binding::specs25::budget10' \
    --output_dir outputs/full \
    --target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
    --num_train_epochs 3 \
    --lora_r 16 \
    --lora_alpha 32 \
    --seed 11 \
    --torch_dtype bfloat16 \
    --localization_condition full

Early layers:
  python src/train_fullstack_lora.py \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --run_id 'calib::lexical_binding::specs25::budget10' \
    --output_dir outputs/early \
    --target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
    --num_train_epochs 3 \
    --lora_r 16 \
    --lora_alpha 32 \
    --seed 11 \
    --torch_dtype bfloat16 \
    --localization_condition early \
    --region_width 8
"""

from __future__ import annotations

import argparse
import inspect
import json
import random
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    set_seed,
)
from peft import LoraConfig, TaskType, get_peft_model

from common import read_jsonl, load_examples_by_id, format_prompt


class SupervisedCompletionDataset(Dataset):
    def __init__(self, examples: List[Dict[str, Any]], tokenizer, max_length: int = 512):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        e = self.examples[idx]
        prompt = format_prompt(e["prompt"])
        answer = e["target"].strip() + self.tokenizer.eos_token

        # This intentionally matches the original full-stack calibration trainer.
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        answer_ids = self.tokenizer(answer, add_special_tokens=False)["input_ids"]

        input_ids = (prompt_ids + answer_ids)[: self.max_length]
        labels = ([-100] * len(prompt_ids) + answer_ids)[: self.max_length]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.ones(len(input_ids), dtype=torch.long),
        }


class DataCollator:
    def __init__(self, tokenizer, pad_to_multiple_of=8):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features):
        max_len = max(len(f["input_ids"]) for f in features)
        if self.pad_to_multiple_of:
            m = self.pad_to_multiple_of
            max_len = ((max_len + m - 1) // m) * m

        batch = {}
        for key in ["input_ids", "attention_mask", "labels"]:
            padded = []
            for f in features:
                x = f[key]
                pad_len = max_len - len(x)
                pad_val = (
                    self.tokenizer.pad_token_id
                    if key == "input_ids"
                    else (-100 if key == "labels" else 0)
                )
                padded.append(torch.cat([x, torch.full((pad_len,), pad_val, dtype=x.dtype)]))
            batch[key] = torch.stack(padded)
        return batch


def parse_layer_indices(s: Optional[str]) -> Optional[List[int]]:
    if not s:
        return None

    layers: List[int] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            layers.extend(range(int(a), int(b) + 1))
        else:
            layers.append(int(part))

    return sorted(set(layers))


def infer_num_layers(model) -> int:
    cfg = model.config
    if hasattr(cfg, "num_hidden_layers"):
        return int(cfg.num_hidden_layers)
    if hasattr(cfg, "n_layer"):
        return int(cfg.n_layer)
    if hasattr(cfg, "num_layers"):
        return int(cfg.num_layers)
    raise ValueError("Could not infer number of transformer layers from model config.")


def layers_for_condition(
    condition: str,
    num_layers: int,
    region_width: int,
    explicit_layers: Optional[List[int]],
) -> Optional[List[int]]:
    condition = condition.lower()

    if condition == "full":
        return None

    if condition == "layers":
        if not explicit_layers:
            raise ValueError("--layer_indices is required when --localization_condition layers")
        bad = [x for x in explicit_layers if x < 0 or x >= num_layers]
        if bad:
            raise ValueError(f"Layer indices out of range for {num_layers} layers: {bad}")
        return explicit_layers

    width = min(region_width, num_layers)

    if condition == "early":
        start = 0
    elif condition == "middle":
        start = (num_layers - width) // 2
    elif condition == "late":
        start = num_layers - width
    else:
        raise ValueError(
            f"Unknown localization_condition={condition}. "
            "Use full, early, middle, late, or layers."
        )

    return list(range(start, start + width))


def dtype_from_arg(arg):
    if arg == "auto":
        return "auto"
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[arg]


def trainable_parameter_report(model) -> Dict[str, Any]:
    trainable = 0
    total = 0
    for _, p in model.named_parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n

    return {
        "trainable_parameters": int(trainable),
        "total_parameters": int(total),
        "trainable_fraction": float(trainable / total) if total else None,
    }


def make_training_arguments(**kwargs):
    """Build TrainingArguments across Transformers versions.

    Preserve the requested training semantics when argument names differ
    across Transformers versions. In particular, if ``warmup_ratio`` is not
    supported but ``warmup_steps`` is, pass the ratio as a float
    ``warmup_steps`` value rather than silently dropping warmup.
    """

    supported = set(inspect.signature(TrainingArguments.__init__).parameters)
    normalized = dict(kwargs)
    translated: Dict[str, Any] = {}

    if "warmup_ratio" in normalized and "warmup_ratio" not in supported:
        warmup_ratio = normalized.pop("warmup_ratio")
        if "warmup_steps" in supported:
            normalized["warmup_steps"] = float(warmup_ratio)
            translated["warmup_ratio"] = {
                "to": "warmup_steps",
                "value": float(warmup_ratio),
            }

    filtered = {key: value for key, value in normalized.items() if key in supported}
    dropped = sorted(set(normalized) - set(filtered))

    if translated:
        print(
            "TrainingArguments compatibility: translated arguments "
            f"{translated}"
        )
    if dropped:
        print(
            "TrainingArguments compatibility: dropped unsupported kwargs "
            f"for this Transformers version: {dropped}"
        )

    compatibility = {
        "translated": translated,
        "dropped": dropped,
    }
    return TrainingArguments(**filtered), compatibility


def parse_args():
    p = argparse.ArgumentParser()

    # Original arguments
    p.add_argument("--model_name", required=True)
    p.add_argument("--examples_path", default="data/prompt_examples.jsonl")
    p.add_argument("--manifest_path", default="data/calibration_manifest.jsonl")
    p.add_argument("--run_id", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--target_modules", nargs="+", required=True)
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--num_train_epochs", type=float, default=2.0)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument(
        "--save_strategy",
        default=None,
        choices=["no", "steps", "epoch"],
        help=(
            "Trainer checkpoint strategy. If omitted, compiler:: runs default "
            "to 'no' to avoid intermediate checkpoints; other runs default to "
            "'epoch' for backward compatibility."
        ),
    )
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--device_map", default="auto")
    p.add_argument("--torch_dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])

    # Seed support
    p.add_argument("--seed", type=int, default=1)

    # Localization additions
    p.add_argument(
        "--localization_condition",
        default="full",
        choices=["full", "early", "middle", "late", "layers"],
        help="Where to insert LoRA modules. 'full' matches the original full-stack trainer.",
    )
    p.add_argument(
        "--region_width",
        type=int,
        default=8,
        help="Number of contiguous layers for early/middle/late localization.",
    )
    p.add_argument(
        "--layer_indices",
        default=None,
        help="Explicit layers for --localization_condition layers, e.g. '0-3,12,13'.",
    )
    p.add_argument(
        "--layers_pattern",
        default="layers",
        help="PEFT layers_pattern. For Llama-family models this is usually 'layers'.",
    )
    p.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Optional. Leave off if you want the closest match to prior calibration.",
    )

    return p.parse_args()


def main():
    args = parse_args()

    set_seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    examples_by_id = load_examples_by_id(args.examples_path)
    manifest_rows = read_jsonl(args.manifest_path)
    run = next((r for r in manifest_rows if r["run_id"] == args.run_id), None)
    if run is None:
        raise ValueError(f"Could not find run_id={args.run_id}")

    # Preserve the historical epoch-checkpoint behavior for legacy runs while
    # making compiler jobs disk-safe by default. Compiler adapters are transient:
    # the runner evaluates the final adapter and may delete it afterward.
    resolved_save_strategy = args.save_strategy
    if resolved_save_strategy is None:
        resolved_save_strategy = (
            "no" if str(args.run_id).startswith("compiler::") else "epoch"
        )

    train_examples = [examples_by_id[eid] for eid in run["train_example_ids"]]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # A failed previous compiler attempt may have left a partially written
    # Trainer checkpoint. When checkpoints are disabled, remove only those
    # stale checkpoint-* directories inside this job's output directory.
    if resolved_save_strategy == "no":
        stale_checkpoints = sorted(
            path for path in output_dir.glob("checkpoint-*") if path.is_dir()
        )
        for checkpoint_dir in stale_checkpoints:
            print(f"Removing stale checkpoint directory: {checkpoint_dir}")
            shutil.rmtree(checkpoint_dir)

    (output_dir / "run_manifest.json").write_text(json.dumps(run, indent=2), encoding="utf-8")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map=args.device_map,
        torch_dtype=dtype_from_arg(args.torch_dtype),
    )
    model.config.use_cache = False

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    num_layers = infer_num_layers(model)
    selected_layers = layers_for_condition(
        args.localization_condition,
        num_layers=num_layers,
        region_width=args.region_width,
        explicit_layers=parse_layer_indices(args.layer_indices),
    )

    lora_kwargs = dict(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        task_type=TaskType.CAUSAL_LM,
        target_modules=args.target_modules,
    )

    # This is the only difference between full-stack and localized conditions.
    if selected_layers is not None:
        lora_kwargs["layers_to_transform"] = selected_layers
        lora_kwargs["layers_pattern"] = args.layers_pattern

    lora_cfg = LoraConfig(**lora_kwargs)
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    param_report = trainable_parameter_report(model)

    training_args, training_compatibility = make_training_arguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_strategy=resolved_save_strategy,
        report_to=[],
        remove_unused_columns=False,
        bf16=args.bf16,
        fp16=args.fp16,
        seed=args.seed,
        data_seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=SupervisedCompletionDataset(train_examples, tokenizer, max_length=args.max_length),
        data_collator=DataCollator(tokenizer),
    )

    trainer.train()
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    metadata = {
        "model_name": args.model_name,
        "run_id": args.run_id,
        "learning_type": run.get("learning_type"),
        "n_specs": run.get("n_specs"),
        "train_budget_per_spec": run.get("train_budget_per_spec"),
        "seed": args.seed,
        "localization_condition": args.localization_condition,
        "num_layers": num_layers,
        "region_width": args.region_width,
        "selected_layers": selected_layers if selected_layers is not None else "all",
        "layers_pattern": args.layers_pattern,
        "target_modules": args.target_modules,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "num_train_examples": len(train_examples),
        "requested_warmup_ratio": args.warmup_ratio,
        "requested_save_strategy": args.save_strategy,
        "resolved_save_strategy": resolved_save_strategy,
        "training_arguments_compatibility": training_compatibility,
        "dropped_training_arguments_for_transformers_compatibility": training_compatibility["dropped"],
        **param_report,
    }
    (output_dir / "localization_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(metadata, indent=2))
    print(f"Saved adapter to {args.output_dir}")


if __name__ == "__main__":
    main()
