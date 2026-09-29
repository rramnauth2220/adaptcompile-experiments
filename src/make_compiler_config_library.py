#!/usr/bin/env python3
"""Create the phase-0 adaptation-compiler configuration library."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Sequence

try:  # pragma: no cover - package import path
    from .compiler_common import (
        CONDITIONS,
        DEFAULT_TARGET_MODULES,
        approximate_parameter_cost,
        resolve_layers,
        selected_layer_count,
        write_jsonl,
    )
except ImportError:  # pragma: no cover - script execution path
    from compiler_common import (
        CONDITIONS,
        DEFAULT_TARGET_MODULES,
        approximate_parameter_cost,
        resolve_layers,
        selected_layer_count,
        write_jsonl,
    )


def module_family_from_targets(target_modules: Sequence[str]) -> str:
    attn = {"q_proj", "k_proj", "v_proj", "o_proj"}
    mlp = {"gate_proj", "up_proj", "down_proj"}
    targets = set(target_modules)
    if targets and targets.issubset(attn):
        return "attention"
    if targets and targets.issubset(mlp):
        return "mlp"
    if targets & attn and targets & mlp:
        return "all"
    return "custom"


def build_config_library(
    conditions: Sequence[str] = CONDITIONS,
    target_modules: Sequence[str] = DEFAULT_TARGET_MODULES,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    n_layers: int = 32,
    region_width: int = 8,
    layers_pattern: str = "layers",
    include_capacity_matched_full: bool = True,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    module_family = module_family_from_targets(target_modules)
    for condition in conditions:
        resolved = resolve_layers(condition, n_layers=n_layers, region_width=region_width)
        row: Dict[str, Any] = {
            "schema_version": 1,
            "config_id": f"{condition}__{module_family}__r{lora_r}",
            "localization_condition": condition,
            "module_family": module_family,
            "target_modules": list(target_modules),
            "lora_r": int(lora_r),
            "lora_alpha": int(lora_alpha),
            "lora_dropout": float(lora_dropout),
            "layer_indices": None,
            "resolved_layer_indices": resolved,
            "n_layers": int(n_layers),
            "region_width": int(region_width),
            "layers_pattern": layers_pattern,
            "n_layers_adapted": 0,
            "approximate_rank_layer_product": 0,
            "approximate_parameter_cost": 0,
        }
        row["n_layers_adapted"] = selected_layer_count(row, n_layers=n_layers)
        row["approximate_rank_layer_product"] = row["n_layers_adapted"] * int(lora_r)
        row["approximate_parameter_cost"] = approximate_parameter_cost(row, n_layers=n_layers)
        rows.append(row)

    if include_capacity_matched_full:
        rank_multiplier = max(1.0, float(n_layers) / float(region_width))
        matched_rank = max(1, int(round(float(lora_r) / rank_multiplier)))
        alpha_ratio = float(lora_alpha) / float(lora_r)
        matched_alpha = max(1, int(round(float(matched_rank) * alpha_ratio)))
        matched_config_id = f"full__{module_family}__r{matched_rank}"
        if matched_config_id not in {row["config_id"] for row in rows}:
            row = {
                "schema_version": 1,
                "config_id": matched_config_id,
                "localization_condition": "full",
                "module_family": module_family,
                "target_modules": list(target_modules),
                "lora_r": int(matched_rank),
                "lora_alpha": int(matched_alpha),
                "lora_dropout": float(lora_dropout),
                "layer_indices": None,
                "resolved_layer_indices": None,
                "n_layers": int(n_layers),
                "region_width": int(region_width),
                "layers_pattern": layers_pattern,
                "n_layers_adapted": 0,
                "approximate_rank_layer_product": 0,
                "approximate_parameter_cost": 0,
                "capacity_matched_to": f"localized__{module_family}__r{lora_r}",
                "capacity_matching_note": (
                    "Full-depth LoRA rank scaled by region_width / n_layers "
                    "to match localized rank-layer-module cost."
                ),
            }
            row["n_layers_adapted"] = selected_layer_count(row, n_layers=n_layers)
            row["approximate_rank_layer_product"] = row["n_layers_adapted"] * int(matched_rank)
            row["approximate_parameter_cost"] = approximate_parameter_cost(row, n_layers=n_layers)
            rows.append(row)
    return rows


def validate_config_library(rows: Sequence[Dict[str, Any]]) -> None:
    seen = set()
    for row in rows:
        required = {
            "config_id",
            "localization_condition",
            "module_family",
            "target_modules",
            "lora_r",
            "lora_alpha",
            "layer_indices",
            "resolved_layer_indices",
            "n_layers_adapted",
            "approximate_parameter_cost",
        }
        missing = required - set(row)
        if missing:
            raise ValueError(f"Config row missing fields: {sorted(missing)}")
        if row["config_id"] in seen:
            raise ValueError(f"Duplicate config_id: {row['config_id']}")
        seen.add(row["config_id"])
        if row["localization_condition"] not in {"full", "early", "middle", "late", "layers"}:
            raise ValueError(f"Unknown condition: {row['localization_condition']}")
        if not row["target_modules"]:
            raise ValueError(f"{row['config_id']} has no target modules")
        if int(row["lora_r"]) <= 0 or int(row["lora_alpha"]) <= 0:
            raise ValueError(f"{row['config_id']} has non-positive LoRA rank/alpha")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build compiler configuration library.")
    parser.add_argument("--output", default="data/compiler/config_library.jsonl")
    parser.add_argument("--conditions", nargs="+", default=CONDITIONS, choices=CONDITIONS)
    parser.add_argument("--target_modules", nargs="+", default=DEFAULT_TARGET_MODULES)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--n_layers", type=int, default=32)
    parser.add_argument("--region_width", type=int, default=8)
    parser.add_argument("--layers_pattern", default="layers")
    parser.add_argument(
        "--no_capacity_matched_full",
        action="store_true",
        help="Write only the base condition grid, omitting the matched full-depth rank control.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = build_config_library(
        conditions=args.conditions,
        target_modules=args.target_modules,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        n_layers=args.n_layers,
        region_width=args.region_width,
        layers_pattern=args.layers_pattern,
        include_capacity_matched_full=not args.no_capacity_matched_full,
    )
    validate_config_library(rows)
    write_jsonl(args.output, rows)
    print(f"Wrote {len(rows)} compiler configs to {args.output}")


if __name__ == "__main__":
    main()
