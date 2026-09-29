#!/usr/bin/env python3
"""Shared helpers for adaptation-compiler pilot infrastructure.

This module deliberately mirrors the existing localization constants without
changing the original AAAI scoring or training scripts.
"""

from __future__ import annotations

import csv
import gzip
import json
import math
import os
import re
import shlex
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


OBJECTIVE_BUDGETS = {
    "lexical_binding": 10,
    "factual_association": 8,
    "behavioral_policy": 10,
    "causal_mapping": 10,
    "procedural_reasoning": 8,
}

BOUND_SOURCE = {
    "lexical_binding": "strict_accuracy",
    "factual_association": "strict_accuracy",
    "behavioral_policy": "concept_accuracy",
    "causal_mapping": "concept_accuracy",
    "procedural_reasoning": "concept_accuracy",
}

OBJECTIVE_EVAL_SCRIPTS = {
    "lexical_binding": "src/evaluate_calibration_run.py",
    "factual_association": "src/evaluate_calibration_run.py",
    "behavioral_policy": "src/evaluate_calibration_run_behavioral.py",
    "causal_mapping": "src/evaluate_calibration_run_causal.py",
    "procedural_reasoning": "src/evaluate_calibration_run_procedural.py",
}

CONDITIONS = ["full", "early", "middle", "late"]
SPLITS = ["train", "id_eval", "paraphrase_eval", "generalization", "negative_control"]
METRICS = ["id_eval", "paraphrase_eval", "acquisition", "transfer", "boundedness"]

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


def shell_join(cmd: Sequence[Any]) -> str:
    parts = [str(x) for x in cmd]
    return subprocess.list2cmdline(parts) if os.name == "nt" else shlex.join(parts)


def model_slug(model_name: str) -> str:
    slug = model_name.split("/")[-1].lower().replace(".", "_")
    return re.sub(r"[^a-z0-9]+", "_", slug).strip("_") or "model"


def maybe_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def boolish_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    if isinstance(value, str):
        low = value.strip().lower()
        if low in {"true", "t", "yes"}:
            return 1.0
        if low in {"false", "f", "no"}:
            return 0.0
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    return maybe_float(value)


def metric_value(row: Dict[str, Any], metric: str) -> Optional[float]:
    """Extract a row-level metric without redefining objective scoring.

    The existing evaluators use slightly different aliases across objectives.
    This function only normalizes those aliases.
    """

    if metric == "strict_accuracy":
        keys = ["strict_accuracy", "strict_score", "strict_correct", "strict", "passed"]
    elif metric == "concept_accuracy":
        keys = [
            "concept_accuracy",
            "concept_score",
            "concept_correct",
            "concept",
            "target_mentioned",
        ]
    elif metric == "passed_accuracy":
        keys = ["passed_accuracy", "passed", "passed_correct"]
    else:
        keys = [metric]

    for key in keys:
        if key in row:
            return boolish_float(row[key])
    return None


def metric_source_for(learning_type: str, split: str) -> str:
    if split in {"id_eval", "paraphrase_eval", "generalization"}:
        return "strict_accuracy"
    if split == "negative_control":
        return BOUND_SOURCE[learning_type]
    raise ValueError(f"No metric source defined for split={split!r}")


def mean(values: Iterable[float]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def parse_layer_indices(value: str | None) -> Optional[List[int]]:
    if not value:
        return None
    layers: List[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            layers.extend(range(int(a), int(b) + 1))
        else:
            layers.append(int(part))
    return sorted(set(layers))


def resolve_layers(
    condition: str,
    n_layers: int,
    region_width: int,
    layer_indices: Optional[Sequence[int]] = None,
) -> Optional[List[int]]:
    condition = condition.lower()
    if condition == "full":
        return None
    if condition == "layers":
        if not layer_indices:
            raise ValueError("layer_indices are required for condition='layers'")
        bad = [idx for idx in layer_indices if idx < 0 or idx >= n_layers]
        if bad:
            raise ValueError(f"Layer indices out of range for {n_layers} layers: {bad}")
        return sorted(set(int(idx) for idx in layer_indices))

    width = min(int(region_width), int(n_layers))
    if width <= 0:
        raise ValueError("region_width must be positive")
    if condition == "early":
        start = 0
    elif condition == "middle":
        start = (n_layers - width) // 2
    elif condition == "late":
        start = n_layers - width
    else:
        raise ValueError(f"Unknown localization condition: {condition}")
    return list(range(start, start + width))


def selected_layer_count(config: Dict[str, Any], n_layers: int) -> int:
    layers = config.get("resolved_layer_indices", config.get("layer_indices"))
    condition = config.get("localization_condition", "full")
    if condition == "full" or layers in (None, "all"):
        return int(n_layers)
    return len(layers)


def approximate_parameter_cost(config: Dict[str, Any], n_layers: int) -> int:
    """Rank-layer-module product used as a lightweight cost proxy."""

    rank = int(config.get("lora_r", 0) or 0)
    n_modules = len(config.get("target_modules") or [])
    return rank * selected_layer_count(config, n_layers=n_layers) * n_modules


def balanced_utility(row: Dict[str, Any]) -> Optional[float]:
    vals = [row.get("acquisition"), row.get("transfer"), row.get("boundedness")]
    if any(v in (None, "") for v in vals):
        return None
    if row.get("preservation") not in (None, ""):
        vals.append(row.get("preservation"))
    return sum(float(v) for v in vals) / len(vals)
