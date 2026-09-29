#!/usr/bin/env python3
"""Publication figures for Experiment 6: Gemma compiler replication.

All reported quantities are derived from the experiment CSV outputs passed on
the command line. The script validates the expected episode/config coverage
before plotting, learns fixed baselines from TRAIN rows only, and evaluates all
headline quantities on TEST rows only.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR / "compiler"))

from plot_style import (  # noqa: E402
    N_BOOT,
    OBJECTIVE_LABELS,
    OBJECTIVE_ORDER,
    OUTCOMES,
    PROGRAM_COLORS,
    PROGRAM_LABELS,
    PROGRAM_ORDER,
    SEED,
    SELECTOR_COLORS,
    SELECTOR_LABELS,
    SELECTOR_MARKERS,
    TIE_TOLERANCE,
    bootstrap_ci,
    deterministic_jitter,
    format_number,
    json_script_metadata,
    objective_label,
    program_label,
    read_csv,
    save_outputs,
    setup_matplotlib,
)


GEOMETRY_EXPECTED_EPISODES = {
    "train": 400,
    "validation": 100,
    "test": 100,
}
OBJECTIVE_EXPECTED_EPISODES = {
    "train": 80,
    "validation": 20,
    "test": 20,
}
SELECTOR_ORDER = ["global_fixed", "objective_fixed", "compiler", "oracle"]
OBJECTIVE_COLORS = {
    "behavioral_policy": "#4477AA",
    "causal_mapping": "#EE6677",
    "factual_association": "#228833",
    "lexical_binding": "#AA3377",
    "procedural_reasoning": "#CCBB44",
}
OBJECTIVE_MARKERS = {
    "behavioral_policy": "o",
    "causal_mapping": "s",
    "factual_association": "D",
    "lexical_binding": "^",
    "procedural_reasoning": "v",
}


def require_columns(df: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = sorted(set(columns) - set(df.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def numeric(df: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    for column in columns:
        try:
            df[column] = pd.to_numeric(df[column])
        except Exception as exc:
            raise ValueError(f"{name}.{column} must be numeric") from exc


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


def utility_from_outcomes(df: pd.DataFrame, prefix: str = "") -> pd.Series:
    required = [f"{prefix}{outcome}" for outcome in OUTCOMES]
    require_columns(df, required, "utility table")
    return sum(df[f"{prefix}{outcome}"].astype(float) for outcome in OUTCOMES) / len(OUTCOMES)


def verify_utility(df: pd.DataFrame, utility_column: str, prefix: str = "", tolerance: float = 1e-8) -> None:
    expected = utility_from_outcomes(df, prefix=prefix)
    diff = (df[utility_column].astype(float) - expected).abs().max()
    if diff > tolerance:
        raise ValueError(
            f"{utility_column} does not match balanced A/T/B/P utility; max absolute difference={diff}"
        )


def choice_from_means(means: Mapping[str, float]) -> str:
    best = max(float(means[cfg]) for cfg in PROGRAM_ORDER)
    tied = [cfg for cfg in PROGRAM_ORDER if best - float(means[cfg]) <= TIE_TOLERANCE]
    return tied[0]


def validate_episode_grid(df: pd.DataFrame, split: str) -> None:
    split_df = df[df["meta_split"] == split]
    expected_episodes = GEOMETRY_EXPECTED_EPISODES[split]
    n_episodes = split_df["episode_id"].nunique()
    if n_episodes != expected_episodes:
        raise ValueError(f"Expected {expected_episodes} {split} episodes, found {n_episodes}.")
    expected_rows = expected_episodes * len(PROGRAM_ORDER)
    if len(split_df) != expected_rows:
        raise ValueError(f"Expected {expected_rows} {split} rows, found {len(split_df)}.")

    duplicated = split_df.duplicated(["episode_id", "config_id"]).sum()
    if duplicated:
        raise ValueError(f"{split} geometry has duplicate episode/config rows: {duplicated}")

    expected_configs = set(PROGRAM_ORDER)
    bad_config_rows = split_df[~split_df["config_id"].isin(expected_configs)]
    if not bad_config_rows.empty:
        configs = sorted(bad_config_rows["config_id"].astype(str).unique())
        raise ValueError(f"{split} geometry contains unexpected config_ids: {configs}")

    coverage = split_df.groupby("episode_id")["config_id"].agg(lambda values: set(values))
    bad = {episode: sorted(expected_configs - present) for episode, present in coverage.items() if present != expected_configs}
    if bad:
        first_episode, missing = next(iter(bad.items()))
        raise ValueError(f"{split} episode {first_episode} lacks expected config coverage; missing={missing}")

    objective_counts = split_df.drop_duplicates("episode_id").groupby("learning_type")["episode_id"].count()
    for objective in OBJECTIVE_ORDER:
        count = int(objective_counts.get(objective, 0))
        if count != OBJECTIVE_EXPECTED_EPISODES[split]:
            raise ValueError(
                f"Expected {OBJECTIVE_EXPECTED_EPISODES[split]} {split} episodes for {objective}, found {count}."
            )

    learning_type_per_episode = split_df.groupby("episode_id")["learning_type"].nunique()
    if int((learning_type_per_episode != 1).sum()):
        raise ValueError(f"{split} geometry has episodes with multiple learning_type values.")


def load_geometry(path: Path) -> pd.DataFrame:
    geometry = read_csv(path)
    require_columns(
        geometry,
        [
            "episode_id",
            "learning_type",
            "meta_split",
            "config_id",
            "acquisition",
            "transfer",
            "boundedness",
            "preservation",
            "utility",
        ],
        "geometry_dataset",
    )
    numeric(geometry, [*OUTCOMES, "utility"], "geometry_dataset")
    observed_splits = set(geometry["meta_split"].astype(str).unique())
    expected_splits = set(GEOMETRY_EXPECTED_EPISODES)
    if observed_splits != expected_splits:
        raise ValueError(f"Expected geometry splits {sorted(expected_splits)}, found {sorted(observed_splits)}.")
    if set(geometry["config_id"].astype(str).unique()) != set(PROGRAM_ORDER):
        raise ValueError(
            "Geometry config coverage must exactly match "
            f"{PROGRAM_ORDER}; found {sorted(geometry['config_id'].astype(str).unique())}."
        )
    verify_utility(geometry, "utility")

    split_leakage = geometry.groupby("episode_id")["meta_split"].nunique()
    leaked = split_leakage[split_leakage > 1]
    if not leaked.empty:
        raise ValueError(f"Episode IDs appear in multiple splits: {list(leaked.index[:5])}")
    for split in GEOMETRY_EXPECTED_EPISODES:
        validate_episode_grid(geometry, split)
    return geometry


def load_predictions(path: Path, test_episodes: set[str]) -> pd.DataFrame:
    predictions = read_csv(path)
    require_columns(
        predictions,
        ["episode_id", "learning_type", "config_id", "observed_utility", "primary_predicted_utility"],
        "predictions_test",
    )
    if "meta_split" in predictions.columns:
        bad = sorted(set(predictions["meta_split"].astype(str).unique()) - {"test"})
        if bad:
            raise ValueError(f"predictions_test contains non-test splits: {bad}")
    if "predicted_utility" in predictions.columns:
        numeric(predictions, ["predicted_utility", "primary_predicted_utility"], "predictions_test")
        diff = (predictions["predicted_utility"] - predictions["primary_predicted_utility"]).abs().max()
        if diff > 1e-9:
            raise ValueError(
                "predicted_utility and primary_predicted_utility disagree; "
                f"max absolute difference={diff}"
            )
        predictions["_primary_predicted_utility"] = predictions["predicted_utility"].astype(float)
    else:
        numeric(predictions, ["primary_predicted_utility"], "predictions_test")
        predictions["_primary_predicted_utility"] = predictions["primary_predicted_utility"].astype(float)
    numeric(predictions, ["observed_utility"], "predictions_test")

    if len(predictions) != len(test_episodes) * len(PROGRAM_ORDER):
        raise ValueError(f"Expected {len(test_episodes) * len(PROGRAM_ORDER)} prediction rows, found {len(predictions)}.")
    if set(predictions["episode_id"].astype(str).unique()) != test_episodes:
        raise ValueError("predictions_test episode IDs do not exactly match geometry TEST episodes.")
    if set(predictions["config_id"].astype(str).unique()) != set(PROGRAM_ORDER):
        raise ValueError("predictions_test config IDs do not exactly match the four candidate configs.")
    duplicated = predictions.duplicated(["episode_id", "config_id"]).sum()
    if duplicated:
        raise ValueError(f"predictions_test has duplicate episode/config rows: {duplicated}")
    coverage = predictions.groupby("episode_id")["config_id"].agg(lambda values: set(values))
    bad = {episode: sorted(set(PROGRAM_ORDER) - present) for episode, present in coverage.items() if present != set(PROGRAM_ORDER)}
    if bad:
        first_episode, missing = next(iter(bad.items()))
        raise ValueError(f"Prediction episode {first_episode} lacks expected config coverage; missing={missing}")
    return predictions


def oracle_by_episode(test: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for episode_id, group in test.groupby("episode_id", sort=True):
        group = group.sort_values("config_id")
        best = float(group["utility"].max())
        oracle_configs = sorted(group.loc[group["utility"] >= best - TIE_TOLERANCE, "config_id"].astype(str))
        ordered = sorted(group["utility"].astype(float), reverse=True)
        top2_margin = ordered[0] - ordered[1] if len(ordered) >= 2 else 0.0
        rows.append(
            {
                "episode_id": str(episode_id),
                "learning_type": str(group["learning_type"].iloc[0]),
                "oracle_utility": best,
                "oracle_config_set": "|".join(oracle_configs),
                "oracle_set_size": len(oracle_configs),
                "top2_observed_margin": top2_margin,
            }
        )
    return pd.DataFrame(rows)


def oracle_set_for(row: Mapping[str, Any]) -> set[str]:
    return {cfg for cfg in str(row["oracle_config_set"]).split("|") if cfg}


def learn_train_defaults(geometry: pd.DataFrame) -> tuple[str, Dict[str, str], Dict[str, Any]]:
    train = geometry[geometry["meta_split"] == "train"].copy()
    global_means = train.groupby("config_id")["utility"].mean().to_dict()
    if set(global_means) != set(PROGRAM_ORDER):
        raise ValueError("TRAIN rows do not cover all candidate configs.")
    global_default = choice_from_means(global_means)

    objective_defaults: Dict[str, str] = {}
    objective_means: Dict[str, Dict[str, float]] = {}
    for objective in OBJECTIVE_ORDER:
        group = train[train["learning_type"] == objective]
        if group.empty:
            raise ValueError(f"No TRAIN rows for {objective}.")
        means = group.groupby("config_id")["utility"].mean().to_dict()
        if set(means) != set(PROGRAM_ORDER):
            raise ValueError(f"TRAIN rows for {objective} do not cover all candidate configs.")
        objective_defaults[objective] = choice_from_means(means)
        objective_means[objective] = {cfg: float(means[cfg]) for cfg in PROGRAM_ORDER}

    metadata = {
        "global_train_mean_utility_by_config": {cfg: float(global_means[cfg]) for cfg in PROGRAM_ORDER},
        "objective_train_mean_utility_by_config": objective_means,
    }
    return global_default, objective_defaults, metadata


def observed_topk_by_episode(test: pd.DataFrame, k: int = 2) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for episode_id, group in test.groupby("episode_id", sort=True):
        observed = group.set_index("config_id")["utility"].astype(float)
        ordered = observed.sort_values(ascending=False)
        threshold = float(ordered.iloc[min(k, len(ordered)) - 1])
        topk_set = sorted([str(cfg) for cfg, value in observed.items() if float(value) >= threshold - TIE_TOLERANCE])
        rows.append(
            {
                "episode_id": str(episode_id),
                "top_k": k,
                "top_k_config_ids": "|".join(topk_set),
            }
        )
    return pd.DataFrame(rows)


def load_compiler_eval(path: Path, test: pd.DataFrame, oracle: pd.DataFrame) -> pd.DataFrame:
    compiler = read_csv(path)
    require_columns(
        compiler,
        [
            "episode_id",
            "selected_config_id",
            "selected_observed_utility",
            "oracle_observed_utility",
            "oracle_regret",
            "oracle_recovery",
            "top_k_recovery",
            "top_k",
        ],
        "compiler_evaluation",
    )
    numeric(
        compiler,
        ["selected_observed_utility", "oracle_observed_utility", "oracle_regret", "top_k"],
        "compiler_evaluation",
    )
    if len(compiler) != 100 or compiler["episode_id"].nunique() != 100:
        raise ValueError(
            f"compiler_evaluation must contain exactly 100 held-out episodes; "
            f"found rows={len(compiler)} episodes={compiler['episode_id'].nunique()}."
        )
    test_episodes = set(test["episode_id"].astype(str).unique())
    if set(compiler["episode_id"].astype(str).unique()) != test_episodes:
        raise ValueError("compiler_evaluation episode IDs do not exactly match geometry TEST episodes.")
    if not set(compiler["selected_config_id"].astype(str).unique()).issubset(set(PROGRAM_ORDER)):
        raise ValueError("compiler_evaluation selected_config_id contains a non-candidate config.")

    utility_lookup = test.set_index(["episode_id", "config_id"])["utility"].astype(float).to_dict()
    learning_lookup = test.drop_duplicates("episode_id").set_index("episode_id")["learning_type"].astype(str).to_dict()
    oracle_lookup = oracle.set_index("episode_id").to_dict(orient="index")
    top2 = observed_topk_by_episode(test, k=2).set_index("episode_id")
    enriched_rows: List[Dict[str, Any]] = []
    for _, row in compiler.iterrows():
        episode_id = str(row["episode_id"])
        selected = str(row["selected_config_id"])
        observed_selected = float(utility_lookup[(episode_id, selected)])
        if abs(observed_selected - float(row["selected_observed_utility"])) > 1e-8:
            raise ValueError(f"Compiler selected utility disagrees with geometry for {episode_id}/{selected}.")
        oracle_row = oracle_lookup[episode_id]
        if abs(float(oracle_row["oracle_utility"]) - float(row["oracle_observed_utility"])) > 1e-8:
            raise ValueError(f"Compiler oracle utility disagrees with geometry for {episode_id}.")
        expected_recovery = selected in oracle_set_for(oracle_row)
        if parse_bool(row["oracle_recovery"]) != expected_recovery:
            raise ValueError(f"compiler_evaluation oracle_recovery disagrees with tie-aware geometry for {episode_id}.")
        top2_row = top2.loc[episode_id]
        observed_top2 = {cfg for cfg in str(top2_row["top_k_config_ids"]).split("|") if cfg}
        if int(row["top_k"]) == 2:
            saved_top2 = {cfg for cfg in str(row["top_k_oracle_config_ids"]).split("|") if cfg}
            if saved_top2 != observed_top2:
                raise ValueError(f"compiler_evaluation top_k_oracle_config_ids disagrees with geometry for {episode_id}.")
            if parse_bool(row["top_k_recovery"]) != (selected in observed_top2):
                raise ValueError(f"compiler_evaluation top_k_recovery disagrees with geometry for {episode_id}.")
        enriched = dict(row)
        enriched["episode_id"] = episode_id
        enriched["learning_type"] = learning_lookup[episode_id]
        enriched["selected_config_id"] = selected
        enriched["tie_aware_oracle_recovery"] = expected_recovery
        enriched["computed_top2_recovery"] = selected in observed_top2
        enriched["computed_top2_config_ids"] = str(top2_row["top_k_config_ids"])
        enriched_rows.append(enriched)
    return pd.DataFrame(enriched_rows)


def build_policy_table(geometry: pd.DataFrame, compiler: pd.DataFrame) -> pd.DataFrame:
    test = geometry[geometry["meta_split"] == "test"].copy()
    global_default, objective_defaults, defaults_metadata = learn_train_defaults(geometry)
    oracle = oracle_by_episode(test)
    compiler_by_episode = compiler.set_index("episode_id")
    rows: List[Dict[str, Any]] = []
    for episode_id, group in test.groupby("episode_id", sort=True):
        learning_type = str(group["learning_type"].iloc[0])
        utilities = group.set_index("config_id")["utility"].astype(float).to_dict()
        oracle_row = oracle[oracle["episode_id"] == str(episode_id)].iloc[0].to_dict()
        oracle_set = oracle_set_for(oracle_row)
        objective_default = objective_defaults[learning_type]
        compiler_row = compiler_by_episode.loc[str(episode_id)]
        selected = str(compiler_row["selected_config_id"])
        oracle_utility = float(oracle_row["oracle_utility"])
        row = {
            "episode_id": str(episode_id),
            "learning_type": learning_type,
            "global_fixed_config_id": global_default,
            "objective_fixed_config_id": objective_default,
            "compiler_config_id": selected,
            "oracle_config_set": oracle_row["oracle_config_set"],
            "oracle_set_size": int(oracle_row["oracle_set_size"]),
            "global_fixed_utility": float(utilities[global_default]),
            "objective_fixed_utility": float(utilities[objective_default]),
            "compiler_utility": float(utilities[selected]),
            "oracle_utility": oracle_utility,
            "global_fixed_oracle_recovery": global_default in oracle_set,
            "objective_fixed_oracle_recovery": objective_default in oracle_set,
            "compiler_oracle_recovery": selected in oracle_set,
        }
        for selector in SELECTOR_ORDER:
            utility_key = f"{selector}_utility"
            row[f"{selector}_regret"] = 0.0 if selector == "oracle" else oracle_utility - float(row[utility_key])
        rows.append(row)
    out = pd.DataFrame(rows)
    out.attrs["global_default"] = global_default
    out.attrs["objective_defaults"] = objective_defaults
    out.attrs["defaults_metadata"] = defaults_metadata
    return out


def selector_summary(policy: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for selector in SELECTOR_ORDER:
        utility_values = policy[f"{selector}_utility"].astype(float).to_numpy()
        regret_values = policy[f"{selector}_regret"].astype(float).to_numpy()
        if selector == "oracle":
            mean_regret, ci_low, ci_high = 0.0, 0.0, 0.0
        else:
            mean_regret, ci_low, ci_high = bootstrap_ci(regret_values, n_boot=N_BOOT, seed=SEED)
        recovery_col = f"{selector}_oracle_recovery"
        if recovery_col in policy.columns:
            top1 = float(policy[recovery_col].astype(bool).mean())
        else:
            top1 = 1.0
        rows.append(
            {
                "selector": selector,
                "selector_label": SELECTOR_LABELS[selector],
                "mean_utility": float(np.mean(utility_values)),
                "mean_oracle_regret": float(mean_regret),
                "ci_low": float(ci_low),
                "ci_high": float(ci_high),
                "top1_oracle_recovery": top1,
            }
        )
    return pd.DataFrame(rows)


def write_csv(path: Path, df: pd.DataFrame) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def plot_calibration(calibration: pd.DataFrame, output_dir: Path, inputs: Sequence[Path]) -> List[Path]:
    require_columns(
        calibration,
        ["grad_accum", "acquisition", "transfer", "boundedness", "preservation", "balanced_utility"],
        "calibration_summary",
    )
    numeric(calibration, ["grad_accum", "acquisition", "transfer", "boundedness", "preservation", "balanced_utility"], "calibration_summary")
    rows = calibration.sort_values("grad_accum").copy()
    expected_ga = [1, 2, 4, 8]
    observed_ga = [int(value) for value in rows["grad_accum"].tolist()]
    if observed_ga != expected_ga:
        raise ValueError(f"Expected calibration grad_accum order {expected_ga}, found {observed_ga}.")

    plt = setup_matplotlib()
    fig, ax = plt.subplots(figsize=(3.35, 2.65))
    x = np.arange(len(rows))
    colors = {
        "acquisition": "#4477AA",
        "transfer": "#228833",
        "boundedness": "#EE6677",
        "preservation": "#AA3377",
        "balanced_utility": "#222222",
    }
    labels = {
        "acquisition": "Acquisition",
        "transfer": "Transfer",
        "boundedness": "Boundedness",
        "preservation": "Preservation",
        "balanced_utility": "Balanced utility",
    }
    for metric in ["acquisition", "transfer", "boundedness", "preservation"]:
        ax.plot(x, rows[metric], marker="o", linewidth=1.3, markersize=4.2, color=colors[metric], label=labels[metric])
    ax.plot(
        x,
        rows["balanced_utility"],
        marker="o",
        linewidth=1.0,
        markersize=3.2,
        linestyle="--",
        color=colors["balanced_utility"],
        label=labels["balanced_utility"],
    )
    ax.axhline(0.80, color="#999999", linewidth=0.8, linestyle=":")
    ax.text(len(rows) - 0.04, 0.815, "pre-specified B gate", ha="right", va="bottom", fontsize=7.4, color="#666666")
    ax.axvline(0, color="#444444", linewidth=0.8, linestyle="--", alpha=0.75)
    ax.text(
        0.055,
        0.30,
        "Selected: GA=1",
        rotation=90,
        ha="center",
        va="center",
        fontsize=7.3,
        color="#333333",
        transform=ax.transAxes,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.2},
    )
    ax.set_xticks(x, [str(int(value)) for value in rows["grad_accum"]])
    ax.set_xlabel("Gradient accumulation")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.04)
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.26),
        frameon=False,
        fontsize=7.0,
        ncol=2,
        columnspacing=0.9,
        handlelength=2.0,
    )
    fig.tight_layout()

    paths = [write_csv(output_dir / "exp6_gemma_calibration.csv", rows)]
    metadata = json_script_metadata(__file__, inputs)
    metadata.update({"selected_grad_accum": 1, "boundedness_gate": 0.80})
    paths.extend(save_outputs(fig, output_dir / "exp6_gemma_calibration", metadata))
    plt.close(fig)
    return paths


def plot_oracle_distribution(oracle: pd.DataFrame, output_dir: Path, inputs: Sequence[Path]) -> tuple[pd.DataFrame, List[Path]]:
    n_episodes = oracle["episode_id"].nunique()
    rows: List[Dict[str, Any]] = []
    for cfg in PROGRAM_ORDER:
        fractional_count = 0.0
        membership_count = 0
        unique_winner_count = 0
        for _, row in oracle.iterrows():
            oracle_set = oracle_set_for(row)
            if cfg in oracle_set:
                membership_count += 1
                fractional_count += 1.0 / len(oracle_set)
                if len(oracle_set) == 1:
                    unique_winner_count += 1
        rows.append(
            {
                "config_id": cfg,
                "label": program_label(cfg),
                "fractional_count": fractional_count,
                "fractional_rate": fractional_count / n_episodes,
                "membership_count": membership_count,
                "unique_winner_count": unique_winner_count,
            }
        )
    summary = pd.DataFrame(rows)

    plt = setup_matplotlib()
    fig, ax = plt.subplots(figsize=(3.25, 2.35))
    x = np.arange(len(summary))
    bars = ax.bar(
        x,
        summary["fractional_rate"],
        color=[PROGRAM_COLORS[cfg] for cfg in summary["config_id"]],
        edgecolor="white",
        linewidth=0.7,
    )
    for bar, value in zip(bars, summary["fractional_rate"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015, f"{100 * value:.0f}%", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x, summary["label"])
    ax.set_ylabel("Fraction of test episodes")
    ax.set_ylim(0, max(0.12, float(summary["fractional_rate"].max()) + 0.08))
    ax.yaxis.set_major_formatter(lambda value, _pos: f"{100 * value:.0f}%")
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    fig.tight_layout()

    paths = [write_csv(output_dir / "exp6_gemma_oracle_distribution.csv", summary)]
    metadata = json_script_metadata(__file__, inputs)
    metadata.update({"n_test_episodes": int(n_episodes), "tie_tolerance": TIE_TOLERANCE})
    paths.extend(save_outputs(fig, output_dir / "exp6_gemma_oracle_distribution", metadata))
    plt.close(fig)
    return summary, paths


def plot_policy_regret(policy: pd.DataFrame, compiler_top2: float, output_dir: Path, inputs: Sequence[Path]) -> tuple[pd.DataFrame, List[Path]]:
    summary = selector_summary(policy)
    plt = setup_matplotlib()
    fig, ax = plt.subplots(figsize=(3.45, 2.35))
    x = np.arange(len(SELECTOR_ORDER))
    for i, selector in enumerate(SELECTOR_ORDER):
        row = summary[summary["selector"] == selector].iloc[0]
        color = SELECTOR_COLORS[selector]
        marker = SELECTOR_MARKERS[selector]
        if selector == "oracle":
            ax.scatter(i, row["mean_oracle_regret"], s=50, marker=marker, color=color, zorder=5)
        else:
            ax.errorbar(
                i,
                row["mean_oracle_regret"],
                yerr=[
                    [row["mean_oracle_regret"] - row["ci_low"]],
                    [row["ci_high"] - row["mean_oracle_regret"]],
                ],
                fmt=marker,
                color=color,
                markersize=5.6,
                capsize=3,
                linewidth=1.1,
                zorder=5,
            )
        if selector == "oracle":
            ax.text(i - 0.10, row["mean_oracle_regret"], f"{row['mean_oracle_regret']:.3f}", ha="right", va="center", fontsize=7.8)
        else:
            ax.text(i + 0.08, row["mean_oracle_regret"], f"{row['mean_oracle_regret']:.3f}", ha="left", va="center", fontsize=7.8)
    ax.axhline(0, color="#777777", linewidth=0.8)
    ax.set_xticks(x, ["Global\nfixed", "Objective\nfixed", "Compiler", "Oracle"])
    ax.set_ylabel("Oracle regret")
    ax.set_xlim(-0.25, len(SELECTOR_ORDER) - 0.55)
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    ymax = max(0.035, float(summary["ci_high"].max()) + 0.008)
    ax.set_ylim(-0.002, ymax)
    fig.tight_layout()

    paths = [write_csv(output_dir / "exp6_gemma_policy_regret.csv", summary)]
    metadata = json_script_metadata(__file__, inputs)
    metadata.update({"compiler_top2_oracle_recovery": compiler_top2, "n_test_episodes": int(policy["episode_id"].nunique())})
    paths.extend(save_outputs(fig, output_dir / "exp6_gemma_policy_regret", metadata))
    plt.close(fig)
    return summary, paths


def plot_delta_by_objective(policy: pd.DataFrame, output_dir: Path, inputs: Sequence[Path]) -> tuple[pd.DataFrame, pd.DataFrame, List[Path]]:
    episodes = policy[
        ["episode_id", "learning_type", "objective_fixed_config_id", "compiler_config_id", "objective_fixed_utility", "compiler_utility"]
    ].copy()
    episodes["compiler_minus_objective_fixed"] = episodes["compiler_utility"] - episodes["objective_fixed_utility"]

    summary_rows: List[Dict[str, Any]] = []
    for objective in OBJECTIVE_ORDER:
        vals = episodes.loc[episodes["learning_type"] == objective, "compiler_minus_objective_fixed"].to_numpy(float)
        mean, lo, hi = bootstrap_ci(vals, n_boot=N_BOOT, seed=SEED)
        summary_rows.append(
            {
                "learning_type": objective,
                "objective": objective_label(objective),
                "n_episodes": len(vals),
                "mean_delta": mean,
                "ci_low": lo,
                "ci_high": hi,
                "n_negative": int(np.sum(vals < 0)),
                "n_positive": int(np.sum(vals > 0)),
            }
        )
    summary = pd.DataFrame(summary_rows)

    plt = setup_matplotlib()
    fig, ax = plt.subplots(figsize=(3.7, 2.45))
    for i, objective in enumerate(OBJECTIVE_ORDER):
        subset = episodes[episodes["learning_type"] == objective]
        vals = subset["compiler_minus_objective_fixed"].to_numpy(float)
        jitter = deterministic_jitter(len(vals), width=0.11, seed=SEED + i)
        ax.scatter(np.full(len(vals), i) + jitter, vals, s=16, color="#555555", alpha=0.38, linewidths=0)
        row = summary[summary["learning_type"] == objective].iloc[0]
        ax.errorbar(
            i,
            row["mean_delta"],
            yerr=[[row["mean_delta"] - row["ci_low"]], [row["ci_high"] - row["mean_delta"]]],
            fmt="o",
            color="#000000",
            markersize=5.2,
            capsize=3,
            linewidth=1.1,
            zorder=5,
        )
    ax.axhline(0, color="#777777", linewidth=0.8)
    ax.set_xticks(np.arange(len(OBJECTIVE_ORDER)), [OBJECTIVE_LABELS[obj] for obj in OBJECTIVE_ORDER], rotation=20, ha="right")
    ax.set_ylabel("Compiler gain over objective-fixed utility")
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    fig.tight_layout()

    paths = [
        write_csv(output_dir / "exp6_gemma_compiler_delta_by_objective.csv", summary),
        write_csv(output_dir / "exp6_gemma_compiler_delta_by_objective_episodes.csv", episodes),
        write_csv(output_dir / "exp6_gemma_compiler_delta_by_objective_summary.csv", summary),
    ]
    metadata = json_script_metadata(__file__, inputs)
    metadata.update({"n_test_episodes": int(episodes["episode_id"].nunique())})
    paths.extend(save_outputs(fig, output_dir / "exp6_gemma_compiler_delta_by_objective", metadata))
    plt.close(fig)
    return episodes, summary, paths


def plot_switch_margins(
    predictions: pd.DataFrame,
    policy: pd.DataFrame,
    geometry: pd.DataFrame,
    output_dir: Path,
    inputs: Sequence[Path],
) -> tuple[pd.DataFrame, List[Path]]:
    test = geometry[geometry["meta_split"] == "test"].copy()
    observed_lookup = test.set_index(["episode_id", "config_id"])["utility"].astype(float).to_dict()
    pred_lookup = predictions.set_index(["episode_id", "config_id"])["_primary_predicted_utility"].astype(float).to_dict()
    rows: List[Dict[str, Any]] = []
    for _, row in policy.iterrows():
        episode_id = str(row["episode_id"])
        learning_type = str(row["learning_type"])
        objective_default = str(row["objective_fixed_config_id"])
        selected = str(row["compiler_config_id"])
        if selected == objective_default:
            continue
        predicted_default = float(pred_lookup[(episode_id, objective_default)])
        predicted_selected = float(pred_lookup[(episode_id, selected)])
        observed_default = float(observed_lookup[(episode_id, objective_default)])
        observed_selected = float(observed_lookup[(episode_id, selected)])
        rows.append(
            {
                "episode_id": episode_id,
                "learning_type": learning_type,
                "objective_default_config": objective_default,
                "selected_config": selected,
                "predicted_default_utility": predicted_default,
                "predicted_selected_utility": predicted_selected,
                "predicted_gain": predicted_selected - predicted_default,
                "observed_default_utility": observed_default,
                "observed_selected_utility": observed_selected,
                "realized_gain": observed_selected - observed_default,
            }
        )
    switches = pd.DataFrame(rows)

    plt = setup_matplotlib()
    fig, ax = plt.subplots(figsize=(3.45, 2.65))
    if switches.empty:
        ax.text(0.5, 0.5, "No compiler departures", ha="center", va="center", transform=ax.transAxes)
    else:
        for objective in OBJECTIVE_ORDER:
            subset = switches[switches["learning_type"] == objective]
            if subset.empty:
                continue
            ax.scatter(
                subset["predicted_gain"],
                subset["realized_gain"],
                s=28,
                marker=OBJECTIVE_MARKERS[objective],
                color=OBJECTIVE_COLORS[objective],
                alpha=0.82,
                edgecolor="white",
                linewidth=0.35,
                label=OBJECTIVE_LABELS[objective],
            )
        frac_negative = float((switches["realized_gain"] < 0).mean())
        ax.text(
            0.02,
            0.98,
            f"switches={len(switches)}\nnegative realized={frac_negative:.2f}",
            ha="left",
            va="top",
            fontsize=7.6,
            transform=ax.transAxes,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.76, "pad": 2.0},
        )
    ax.axvline(0, color="#777777", linewidth=0.8)
    ax.axhline(0, color="#777777", linewidth=0.8)
    ax.set_xlabel("Predicted gain from switching")
    ax.set_ylabel("Realized gain from switching")
    ax.grid(True)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.02), ncol=2, frameon=False, fontsize=7.2, handletextpad=0.3)
    fig.tight_layout()

    paths = [write_csv(output_dir / "exp6_gemma_switch_margins.csv", switches)]
    metadata = json_script_metadata(__file__, inputs)
    metadata.update(
        {
            "n_switches": int(len(switches)),
            "fraction_negative_realized_gain": float((switches["realized_gain"] < 0).mean()) if not switches.empty else 0.0,
        }
    )
    paths.extend(save_outputs(fig, output_dir / "exp6_gemma_switch_margins", metadata))
    plt.close(fig)
    return switches, paths


def plot_utility_heatmap(geometry: pd.DataFrame, output_dir: Path, inputs: Sequence[Path]) -> tuple[pd.DataFrame, List[Path]]:
    test = geometry[geometry["meta_split"] == "test"].copy()
    matrix = test.groupby(["learning_type", "config_id"])["utility"].mean().unstack("config_id")
    matrix = matrix.loc[OBJECTIVE_ORDER, PROGRAM_ORDER]

    plt = setup_matplotlib()
    fig, ax = plt.subplots(figsize=(3.55, 2.65))
    values = matrix.to_numpy(float)
    image = ax.imshow(values, cmap="viridis", aspect="auto", vmin=float(values.min()), vmax=float(values.max()))
    ax.set_xticks(np.arange(len(PROGRAM_ORDER)), [PROGRAM_LABELS[cfg] for cfg in PROGRAM_ORDER])
    ax.set_yticks(np.arange(len(OBJECTIVE_ORDER)), [OBJECTIVE_LABELS[obj] for obj in OBJECTIVE_ORDER])
    ax.tick_params(axis="x", rotation=20)
    midpoint = (float(values.min()) + float(values.max())) / 2.0
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            color = "white" if values[i, j] < midpoint else "black"
            ax.text(j, i, f"{values[i, j]:.3f}", ha="center", va="center", color=color, fontsize=7.7)
    cbar = fig.colorbar(image, ax=ax, fraction=0.045, pad=0.03)
    cbar.set_label("Mean balanced utility")
    fig.tight_layout()

    out_matrix = matrix.reset_index()
    out_matrix.insert(1, "objective", out_matrix["learning_type"].map(OBJECTIVE_LABELS))
    paths = [write_csv(output_dir / "exp6_gemma_utility_heatmap.csv", out_matrix)]
    metadata = json_script_metadata(__file__, inputs)
    metadata.update({"n_test_episodes": int(test["episode_id"].nunique())})
    paths.extend(save_outputs(fig, output_dir / "exp6_gemma_utility_heatmap", metadata))
    plt.close(fig)
    return out_matrix, paths


def compare_existing_summary(compiler_eval: Path, policy_summary: pd.DataFrame, compiler_top2: float) -> List[str]:
    summary_path = compiler_eval.with_name("compiler_evaluation_summary.json")
    if not summary_path.exists():
        return [f"No existing compiler summary found at {summary_path}."]
    existing = json.loads(summary_path.read_text(encoding="utf-8"))
    compiler_row = policy_summary[policy_summary["selector"] == "compiler"].iloc[0]
    oracle_row = policy_summary[policy_summary["selector"] == "oracle"].iloc[0]
    checks = {
        "mean_selected_observed_utility": float(compiler_row["mean_utility"]),
        "mean_oracle_observed_utility": float(oracle_row["mean_utility"]),
        "mean_oracle_regret": float(compiler_row["mean_oracle_regret"]),
        "oracle_recovery_rate": float(compiler_row["top1_oracle_recovery"]),
        "top_k_recovery_rate": float(compiler_top2),
    }
    discrepancies: List[str] = []
    for key, derived in checks.items():
        if key not in existing:
            discrepancies.append(f"{key}: missing from {summary_path}")
            continue
        diff = abs(float(existing[key]) - derived)
        if diff > 1e-9:
            discrepancies.append(f"{key}: derived={derived:.12g}, existing={float(existing[key]):.12g}, diff={diff:.3g}")
    return discrepancies


def build_headline_summary(
    policy_summary: pd.DataFrame,
    oracle_distribution: pd.DataFrame,
    global_default: str,
    objective_defaults: Mapping[str, str],
    compiler_top2: float,
    switches: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = [
        {"section": "defaults", "metric": "global_default", "label": program_label(global_default), "value": global_default},
    ]
    for objective in OBJECTIVE_ORDER:
        cfg = objective_defaults[objective]
        rows.append(
            {
                "section": "defaults",
                "metric": f"objective_default_{objective}",
                "label": f"{OBJECTIVE_LABELS[objective]} default",
                "value": cfg,
            }
        )
    for _, row in policy_summary.iterrows():
        rows.extend(
            [
                {
                    "section": "policy",
                    "metric": f"{row['selector']}_mean_utility",
                    "label": row["selector_label"],
                    "value": row["mean_utility"],
                },
                {
                    "section": "policy",
                    "metric": f"{row['selector']}_mean_oracle_regret",
                    "label": row["selector_label"],
                    "value": row["mean_oracle_regret"],
                },
                {
                    "section": "policy",
                    "metric": f"{row['selector']}_top1_oracle_recovery",
                    "label": row["selector_label"],
                    "value": row["top1_oracle_recovery"],
                },
            ]
        )
    rows.append({"section": "policy", "metric": "compiler_top2_oracle_recovery", "label": "Compiler top-2", "value": compiler_top2})
    rows.append({"section": "switches", "metric": "n_compiler_departures_from_objective_default", "label": "Switches", "value": len(switches)})
    rows.append(
        {
            "section": "switches",
            "metric": "fraction_switches_negative_realized_gain",
            "label": "Negative realized gain",
            "value": float((switches["realized_gain"] < 0).mean()) if not switches.empty else 0.0,
        }
    )
    for _, row in oracle_distribution.iterrows():
        rows.append(
            {
                "section": "oracle_distribution",
                "metric": f"{row['config_id']}_fractional_rate",
                "label": row["label"],
                "value": row["fractional_rate"],
            }
        )
    return pd.DataFrame(rows)


def print_summary(
    policy_summary: pd.DataFrame,
    global_default: str,
    objective_defaults: Mapping[str, str],
    compiler_top2: float,
    switches: pd.DataFrame,
    discrepancies: Sequence[str],
    generated: Sequence[Path],
) -> None:
    by_selector = policy_summary.set_index("selector")
    print("\nExperiment 6 Gemma derived summary")
    print(f"selected global default: {program_label(global_default)} ({global_default})")
    print("objective defaults:")
    for objective in OBJECTIVE_ORDER:
        cfg = objective_defaults[objective]
        print(f"  {OBJECTIVE_LABELS[objective]}: {program_label(cfg)} ({cfg})")
    for selector in ["global_fixed", "objective_fixed", "compiler"]:
        row = by_selector.loc[selector]
        print(
            f"{SELECTOR_LABELS[selector]}: "
            f"utility={format_number(row['mean_utility'])}, "
            f"regret={format_number(row['mean_oracle_regret'])}"
        )
    oracle_row = by_selector.loc["oracle"]
    print(f"Oracle: utility={format_number(oracle_row['mean_utility'])}, regret=0.000")
    compiler_row = by_selector.loc["compiler"]
    print(f"compiler top-1={float(compiler_row['top1_oracle_recovery']):.2f}, top-2={compiler_top2:.2f}")
    print(f"compiler departures from objective defaults: {len(switches)}/100")
    if discrepancies:
        print("existing summary discrepancies:")
        for item in discrepancies:
            print(f"  {item}")
    else:
        print("existing summary discrepancies: none")
    print("generated output paths:")
    for path in generated:
        print(f"  {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Experiment 6 Gemma replication figures.")
    parser.add_argument("--calibration", default="artifacts/gemma/calibration/calibration_summary.csv")
    parser.add_argument("--geometry", default="artifacts/gemma/geometry/geometry_dataset.csv")
    parser.add_argument(
        "--predictions",
        default="artifacts/gemma/prediction/predictions_test.csv",
    )
    parser.add_argument(
        "--compiler-eval",
        default="artifacts/gemma/selection/compiler_evaluation.csv",
    )
    parser.add_argument("--output-dir", default="outputs/release_verification/figures/gemma")
    args = parser.parse_args()

    calibration_path = Path(args.calibration)
    geometry_path = Path(args.geometry)
    predictions_path = Path(args.predictions)
    compiler_eval_path = Path(args.compiler_eval)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    inputs = [calibration_path, geometry_path, predictions_path, compiler_eval_path]

    calibration = read_csv(calibration_path)
    geometry = load_geometry(geometry_path)
    test = geometry[geometry["meta_split"] == "test"].copy()
    test_episodes = set(test["episode_id"].astype(str).unique())
    predictions = load_predictions(predictions_path, test_episodes)
    pred_obs = predictions.set_index(["episode_id", "config_id"])["observed_utility"].astype(float)
    geom_obs = test.set_index(["episode_id", "config_id"])["utility"].astype(float)
    observed_diff = (pred_obs.sort_index() - geom_obs.sort_index()).abs().max()
    if observed_diff > 1e-8:
        raise ValueError(f"predictions_test observed utility disagrees with geometry; max absolute difference={observed_diff}")

    oracle = oracle_by_episode(test)
    compiler = load_compiler_eval(compiler_eval_path, test, oracle)
    policy = build_policy_table(geometry, compiler)
    compiler_top2 = float(compiler["computed_top2_recovery"].astype(bool).mean())

    generated: List[Path] = []
    generated.extend(plot_calibration(calibration, output_dir, inputs))
    oracle_distribution, paths = plot_oracle_distribution(oracle, output_dir, inputs)
    generated.extend(paths)
    policy_summary, paths = plot_policy_regret(policy, compiler_top2, output_dir, inputs)
    generated.extend(paths)
    _delta_episodes, _delta_summary, paths = plot_delta_by_objective(policy, output_dir, inputs)
    generated.extend(paths)
    switches, paths = plot_switch_margins(predictions, policy, geometry, output_dir, inputs)
    generated.extend(paths)
    _heatmap, paths = plot_utility_heatmap(geometry, output_dir, inputs)
    generated.extend(paths)

    global_default = str(policy.attrs["global_default"])
    objective_defaults = dict(policy.attrs["objective_defaults"])
    headline = build_headline_summary(policy_summary, oracle_distribution, global_default, objective_defaults, compiler_top2, switches)
    generated.append(write_csv(output_dir / "exp6_gemma_summary.csv", headline))
    discrepancies = compare_existing_summary(compiler_eval_path, policy_summary, compiler_top2)
    print_summary(policy_summary, global_default, objective_defaults, compiler_top2, switches, discrepancies, generated)


if __name__ == "__main__":
    main()
