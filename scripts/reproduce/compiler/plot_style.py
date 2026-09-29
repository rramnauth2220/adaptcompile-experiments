#!/usr/bin/env python3
"""Shared plotting helpers for compiler paper figures.

The functions here are intentionally deterministic and data-conservative:
fixed-program baselines are learned from TRAIN rows only, uncertainty is
bootstrapped over episodes, and oracle sets are tie-aware.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd


SEED = 2026
N_BOOT = 20_000
TIE_TOLERANCE = 1e-12
OUTCOMES = ["acquisition", "transfer", "boundedness", "preservation"]
OUTCOME_LABELS = {
    "acquisition": "A",
    "transfer": "T",
    "boundedness": "B",
    "preservation": "P",
}

PROGRAM_ORDER = ["early__all__r16", "middle__all__r16", "late__all__r16", "full__all__r4"]
PROGRAM_LABELS = {
    "early__all__r16": "Early",
    "middle__all__r16": "Middle",
    "late__all__r16": "Late",
    "full__all__r4": "Full-r4",
}
PROGRAM_COLORS = {
    "early__all__r16": "#4477AA",
    "middle__all__r16": "#228833",
    "late__all__r16": "#EE6677",
    "full__all__r4": "#CCBB44",
}
PROGRAM_HATCHES = {
    "early__all__r16": "",
    "middle__all__r16": "///",
    "late__all__r16": "\\\\\\",
    "full__all__r4": "...",
}

OBJECTIVE_ORDER = [
    "behavioral_policy",
    "causal_mapping",
    "factual_association",
    "lexical_binding",
    "procedural_reasoning",
]
OBJECTIVE_LABELS = {
    "behavioral_policy": "Behavioral",
    "causal_mapping": "Causal",
    "factual_association": "Factual",
    "lexical_binding": "Lexical",
    "procedural_reasoning": "Procedural",
}
OBJECTIVE_LABELS_COMPACT = {
    "behavioral_policy": "Behavioral",
    "causal_mapping": "Causal",
    "factual_association": "Factual",
    "lexical_binding": "Lexical",
    "procedural_reasoning": "Procedural",
    "macro_average": "Macro",
}

SELECTOR_ORDER = ["global_fixed", "objective_fixed", "compiler", "oracle"]
SELECTOR_LABELS = {
    "global_fixed": "Global fixed",
    "objective_fixed": "Objective fixed",
    "compiler": "Compiler",
    "lofo_compiler": "LOFO compiler",
    "oracle": "Oracle",
    "configuration_mean": "Config. mean",
    "objective_conditioned_mean": "Objective mean",
    "primary": "Primary predictor",
}
SELECTOR_COLORS = {
    "global_fixed": "#777777",
    "objective_fixed": "#66CCEE",
    "compiler": "#AA3377",
    "lofo_compiler": "#AA3377",
    "oracle": "#000000",
    "configuration_mean": "#999999",
    "objective_conditioned_mean": "#66CCEE",
    "primary": "#AA3377",
}
SELECTOR_MARKERS = {
    "global_fixed": "s",
    "objective_fixed": "D",
    "compiler": "o",
    "lofo_compiler": "o",
    "oracle": "*",
    "configuration_mean": "s",
    "objective_conditioned_mean": "D",
    "primary": "o",
}
SELECTOR_LINESTYLES = {
    "global_fixed": "--",
    "objective_fixed": "-.",
    "compiler": "-",
    "lofo_compiler": "-",
    "oracle": ":",
}

UTILITY_SPECS = {
    "balanced": "Balanced",
    "transfer_heavy": "Transfer",
    "boundedness_heavy": "Bounded-\nness",
    "preservation_heavy": "Preser-\nvation",
}
FEATURE_SET_LABELS = {
    "ablation_episode": "Episode",
    "full_auto": "Full",
    "ablation_module_probes": "Module\nprobes",
    "ablation_frozen_behavior": "Frozen\nbehavior",
}


def setup_matplotlib() -> Any:
    cache_dir = Path(tempfile.gettempdir()) / "localized_learning_matplotlib_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.labelsize": 9.5,
            "axes.titlesize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "grid.color": "#D9D9D9",
            "grid.linewidth": 0.6,
            "grid.alpha": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    return plt


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def read_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def read_json(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_dirs(output_root: str | Path) -> Dict[str, Path]:
    root = Path(output_root)
    dirs = {
        "root": root,
        "main": root / "main",
        "appendix": root / "appendix",
        "data": root / "data",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def save_outputs(
    fig: Any,
    stem: Path,
    metadata: Mapping[str, Any],
    dpi: int = 400,
) -> Tuple[Path, Path, Path]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    pdf = stem.with_suffix(".pdf")
    png = stem.with_suffix(".png")
    meta = stem.with_suffix(".metadata.json")
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=dpi, bbox_inches="tight")
    meta.write_text(json.dumps(to_builtin(dict(metadata)), indent=2), encoding="utf-8")
    return pdf, png, meta


def to_builtin(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
    if isinstance(value, pd.Series):
        return value.to_dict()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_summary(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([to_builtin(dict(row)) for row in rows]).to_csv(path, index=False)


def panel_label(ax: Any, label: str) -> None:
    ax.text(
        -0.12,
        1.04,
        label,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def prettify_axes(ax: Any, grid: bool = True) -> None:
    if grid:
        ax.grid(axis="y")
    ax.set_axisbelow(True)


def format_number(value: float | None, digits: int = 3) -> str:
    if value is None or not math.isfinite(float(value)):
        return "NA"
    return f"{float(value):.{digits}f}"


def objective_label(value: str) -> str:
    return OBJECTIVE_LABELS.get(str(value), str(value).replace("_", " ").title())


def program_label(value: str) -> str:
    return PROGRAM_LABELS.get(str(value), str(value))


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "t"}


def weighted_utility(df: pd.DataFrame, weights: Mapping[str, float] | None = None, prefix: str = "") -> pd.Series:
    weights = dict(weights or {metric: 1.0 for metric in OUTCOMES})
    denom = sum(float(weights.get(metric, 0.0)) for metric in OUTCOMES)
    if denom <= 0:
        raise ValueError("Utility weights must sum to a positive value.")
    total = None
    for metric in OUTCOMES:
        column = f"{prefix}{metric}"
        if column not in df.columns:
            raise ValueError(f"Missing utility column: {column}")
        term = df[column].astype(float) * float(weights.get(metric, 0.0))
        total = term if total is None else total + term
    return total / denom


def assert_balanced_utility(df: pd.DataFrame, tolerance: float = 1e-9) -> None:
    if "utility" not in df.columns:
        return
    expected = weighted_utility(df)
    diff = (df["utility"].astype(float) - expected).abs().max()
    if diff > tolerance:
        raise ValueError(f"Balanced utility mismatch: max absolute difference {diff}")


def assert_test_episode_grid(df: pd.DataFrame, n_episodes: int = 100, configs: Sequence[str] = PROGRAM_ORDER) -> None:
    test = df[df["meta_split"] == "test"].copy()
    observed_episodes = test["episode_id"].nunique()
    if observed_episodes != n_episodes:
        raise ValueError(f"Expected {n_episodes} test episodes, found {observed_episodes}.")
    assert_episode_config_grid(test, configs=configs, context="test")


def assert_episode_config_grid(df: pd.DataFrame, configs: Sequence[str] = PROGRAM_ORDER, context: str = "data") -> None:
    duplicated = df.duplicated(["episode_id", "config_id"]).sum()
    if duplicated:
        raise ValueError(f"{context} has duplicate episode/config rows: {duplicated}")
    expected = set(configs)
    coverage = df.groupby("episode_id")["config_id"].agg(lambda vals: set(vals))
    bad = {ep: sorted(expected - present) for ep, present in coverage.items() if present != expected}
    if bad:
        first = next(iter(bad.items()))
        raise ValueError(f"{context} episode lacks four-config coverage: {first}")


def bootstrap_ci(
    values: Sequence[float],
    n_boot: int = N_BOOT,
    seed: int = SEED,
    alpha: float = 0.05,
) -> Tuple[float, float, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    boot = arr[idx].mean(axis=1)
    return (
        float(arr.mean()),
        float(np.quantile(boot, alpha / 2)),
        float(np.quantile(boot, 1 - alpha / 2)),
    )


def deterministic_jitter(n: int, width: float = 0.16, seed: int = SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(-width, width, size=n)


def oracle_sets(
    df: pd.DataFrame,
    weights: Mapping[str, float] | None = None,
    tie_tolerance: float = TIE_TOLERANCE,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    work = df.copy()
    work["_utility"] = weighted_utility(work, weights=weights)
    for episode_id, group in work.groupby("episode_id", sort=True):
        group = group.sort_values("config_id")
        best = float(group["_utility"].max())
        oracle = group[group["_utility"] >= best - tie_tolerance].copy()
        ordered = sorted(group["_utility"].astype(float), reverse=True)
        margin = ordered[0] - ordered[1] if len(ordered) >= 2 else 0.0
        rows.append(
            {
                "episode_id": episode_id,
                "learning_type": str(group["learning_type"].iloc[0]),
                "oracle_utility": best,
                "oracle_config_set": "|".join(sorted(oracle["config_id"].astype(str))),
                "oracle_set_size": int(len(oracle)),
                "top2_margin": float(margin),
            }
        )
    return pd.DataFrame(rows)


def fractional_oracle_winner_shares(oracle_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for _, row in oracle_df.iterrows():
        configs = [cfg for cfg in str(row["oracle_config_set"]).split("|") if cfg]
        credit = 1.0 / len(configs) if configs else 0.0
        for cfg in configs:
            rows.append(
                {
                    "episode_id": row["episode_id"],
                    "learning_type": row["learning_type"],
                    "config_id": cfg,
                    "credit": credit,
                }
            )
    share = pd.DataFrame(rows)
    if share.empty:
        return pd.DataFrame(columns=["learning_type", "config_id", "share"])
    out = (
        share.groupby(["learning_type", "config_id"], as_index=False)["credit"]
        .sum()
        .rename(columns={"credit": "fractional_count"})
    )
    n_by_type = oracle_df.groupby("learning_type")["episode_id"].nunique().to_dict()
    out["share"] = out.apply(lambda r: r["fractional_count"] / n_by_type[str(r["learning_type"])], axis=1)
    return out


def learn_fixed_programs(
    geometry: pd.DataFrame,
    weights: Mapping[str, float] | None = None,
    tie_tolerance: float = TIE_TOLERANCE,
) -> Tuple[str, Dict[str, str], Dict[str, Any]]:
    train = geometry[geometry["meta_split"] == "train"].copy()
    if train.empty:
        raise ValueError("No TRAIN rows available for fixed-program baselines.")
    train["_utility"] = weighted_utility(train, weights=weights)
    mean_by_config = train.groupby("config_id")["_utility"].mean().to_dict()
    best = max(mean_by_config.values())
    global_tie = sorted(cfg for cfg, val in mean_by_config.items() if best - float(val) <= tie_tolerance)
    objective_map: Dict[str, str] = {}
    objective_means: Dict[str, Dict[str, float]] = {}
    for learning_type, group in train.groupby("learning_type"):
        means = group.groupby("config_id")["_utility"].mean().to_dict()
        objective_means[str(learning_type)] = {str(k): float(v) for k, v in means.items()}
        obj_best = max(means.values())
        objective_map[str(learning_type)] = sorted(
            cfg for cfg, val in means.items() if obj_best - float(val) <= tie_tolerance
        )[0]
    return (
        global_tie[0],
        objective_map,
        {
            "global_train_mean_utility_by_config": {str(k): float(v) for k, v in mean_by_config.items()},
            "global_train_tie_set": global_tie,
            "objective_train_mean_utility_by_config": objective_means,
        },
    )


def episode_selector_utilities(
    geometry: pd.DataFrame,
    weights: Mapping[str, float] | None = None,
    compiler_csv: str | Path | None = None,
) -> pd.DataFrame:
    test = geometry[geometry["meta_split"] == "test"].copy()
    assert_episode_config_grid(test, context="test")
    test["_utility"] = weighted_utility(test, weights=weights)
    global_cfg, objective_map, baseline_meta = learn_fixed_programs(geometry, weights=weights)
    oracle = oracle_sets(test, weights=weights)
    rows: List[Dict[str, Any]] = []
    compiler_map: Dict[str, Tuple[str, float]] = {}
    if compiler_csv is not None and Path(compiler_csv).exists():
        comp = read_csv(compiler_csv)
        for _, row in comp.iterrows():
            compiler_map[str(row["episode_id"])] = (
                str(row["selected_config_id"]),
                float(row["selected_observed_utility"]),
            )
    for episode_id, group in test.groupby("episode_id", sort=True):
        learning_type = str(group["learning_type"].iloc[0])
        by_config = group.set_index("config_id")["_utility"].to_dict()
        oracle_row = oracle[oracle["episode_id"] == episode_id].iloc[0]
        compiler_cfg, compiler_u = compiler_map.get(episode_id, ("", float("nan")))
        rows.append(
            {
                "episode_id": episode_id,
                "learning_type": learning_type,
                "global_fixed_config_id": global_cfg,
                "objective_fixed_config_id": objective_map[learning_type],
                "compiler_config_id": compiler_cfg,
                "global_fixed_utility": float(by_config[global_cfg]),
                "objective_fixed_utility": float(by_config[objective_map[learning_type]]),
                "compiler_utility": compiler_u,
                "oracle_utility": float(oracle_row["oracle_utility"]),
                "oracle_config_set": oracle_row["oracle_config_set"],
                "oracle_set_size": int(oracle_row["oracle_set_size"]),
                "global_fixed_regret": float(oracle_row["oracle_utility"] - by_config[global_cfg]),
                "objective_fixed_regret": float(oracle_row["oracle_utility"] - by_config[objective_map[learning_type]]),
                "compiler_regret": float(oracle_row["oracle_utility"] - compiler_u) if compiler_cfg else float("nan"),
            }
        )
    out = pd.DataFrame(rows)
    out.attrs["baseline_metadata"] = baseline_meta
    return out


def prediction_selected_utilities(predictions: pd.DataFrame, method_prefix: str = "primary") -> pd.DataFrame:
    required = {"episode_id", "learning_type", "config_id", "observed_utility", f"{method_prefix}_predicted_utility"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Prediction table missing columns: {sorted(missing)}")
    rows: List[Dict[str, Any]] = []
    for episode_id, group in predictions.groupby("episode_id", sort=True):
        group = group.sort_values("config_id")
        observed = group.set_index("config_id")["observed_utility"].astype(float).to_dict()
        pred = group.set_index("config_id")[f"{method_prefix}_predicted_utility"].astype(float)
        best_pred = pred.max()
        selected = sorted(pred[pred >= best_pred - TIE_TOLERANCE].index.astype(str))[0]
        best_obs = max(observed.values())
        oracle_set = sorted(cfg for cfg, value in observed.items() if best_obs - value <= TIE_TOLERANCE)
        rows.append(
            {
                "episode_id": episode_id,
                "learning_type": str(group["learning_type"].iloc[0]),
                "selected_config_id": selected,
                "selected_observed_utility": float(observed[selected]),
                "oracle_observed_utility": float(best_obs),
                "oracle_regret": float(best_obs - observed[selected]),
                "oracle_recovery": selected in set(oracle_set),
                "oracle_config_set": "|".join(oracle_set),
            }
        )
    return pd.DataFrame(rows)


def json_script_metadata(script_name: str, inputs: Sequence[str | Path], seed: int = SEED) -> Dict[str, Any]:
    return {
        "script": script_name,
        "input_files": [str(path) for path in inputs],
        "git_commit": git_commit(),
        "random_seed": seed,
        "bootstrap_resamples": N_BOOT,
    }
