#!/usr/bin/env python3
"""Plot the adaptation-compilation scope synthesis figure.

This figure intentionally reads the current experiment CSV/JSON outputs. Llama
outputs are authoritative under ``artifacts/llama`` even though the filenames
were reused after causal rescoring; Gemma outputs are read from the explicit
``corrected_causal_v1`` tree.
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
    PROGRAM_ORDER,
    SEED,
    SELECTOR_COLORS,
    TIE_TOLERANCE,
    bootstrap_ci,
    format_number,
    json_script_metadata,
    read_csv,
    save_outputs,
    setup_matplotlib,
)


REPRESENTATION_DIRS = [
    "ablation_episode",
    "full_auto",
    "ablation_module_probes",
    "ablation_frozen_behavior",
]
RELEASE_REPRESENTATION_PATHS = {
    "ablation_episode": Path("ablations/episode"),
    "full_auto": Path("prediction"),
    "ablation_module_probes": Path("ablations/module_probes"),
    "ablation_frozen_behavior": Path("ablations/frozen_behavior"),
}
REPRESENTATION_METHODS = {
    "ablation_episode": "Episode",
    "full_auto": "Full",
    "ablation_module_probes": "Module probes",
    "ablation_frozen_behavior": "Frozen behavior",
}
REPRESENTATION_TICKS = {
    "Episode": "Episode",
    "Full": "Full",
    "Module probes": "Module\nprobes",
    "Frozen behavior": "Frozen\nbehavior",
}

GLOBAL_COLOR = SELECTOR_COLORS.get("global_fixed", "#777777")
OBJECTIVE_COLOR = SELECTOR_COLORS.get("objective_fixed", "#66CCEE")
COMPILER_COLOR = SELECTOR_COLORS.get("compiler", "#AA3377")
NEUTRAL_COLOR = "#777777"


def require_columns(df: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = sorted(set(columns) - set(df.columns))
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")


def ensure_numeric(df: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    for column in columns:
        try:
            df[column] = pd.to_numeric(df[column])
        except Exception as exc:
            raise ValueError(f"{name}.{column} must be numeric") from exc


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]] | pd.DataFrame) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(rows, pd.DataFrame):
        rows.to_csv(path, index=False)
    else:
        pd.DataFrame(rows).to_csv(path, index=False)
    return path


def balanced_utility(df: pd.DataFrame, prefix: str = "") -> pd.Series:
    cols = [f"{prefix}{metric}" for metric in OUTCOMES]
    require_columns(df, cols, "balanced utility input")
    return sum(df[col].astype(float) for col in cols) / len(cols)


def verify_utility(df: pd.DataFrame, *, utility_column: str, prefix: str = "", name: str, tol: float = 1e-8) -> None:
    require_columns(df, [utility_column], name)
    expected = balanced_utility(df, prefix=prefix)
    diff = (df[utility_column].astype(float) - expected).abs().max()
    if diff > tol:
        raise ValueError(f"{name}: {utility_column} differs from balanced A/T/B/P utility; max diff={diff}.")


def validate_four_config_grid(df: pd.DataFrame, *, n_episodes: int, name: str) -> None:
    if df["episode_id"].nunique() != n_episodes:
        raise ValueError(f"{name}: expected {n_episodes} episodes, found {df['episode_id'].nunique()}.")
    expected_rows = n_episodes * len(PROGRAM_ORDER)
    if len(df) != expected_rows:
        raise ValueError(f"{name}: expected {expected_rows} rows, found {len(df)}.")
    duplicated = int(df.duplicated(["episode_id", "config_id"]).sum())
    if duplicated:
        raise ValueError(f"{name}: duplicate episode/config rows: {duplicated}.")
    configs = set(df["config_id"].astype(str).unique())
    if configs != set(PROGRAM_ORDER):
        raise ValueError(f"{name}: expected configs {PROGRAM_ORDER}, found {sorted(configs)}.")
    coverage = df.groupby("episode_id")["config_id"].agg(lambda values: set(values))
    bad = {ep: sorted(set(PROGRAM_ORDER) - present) for ep, present in coverage.items() if present != set(PROGRAM_ORDER)}
    if bad:
        ep, missing = next(iter(bad.items()))
        raise ValueError(f"{name}: incomplete config coverage for {ep}; missing={missing}.")


def load_geometry(path: Path, *, name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    geometry = read_csv(path)
    require_columns(
        geometry,
        ["episode_id", "learning_type", "meta_split", "config_id", *OUTCOMES, "utility"],
        name,
    )
    ensure_numeric(geometry, [*OUTCOMES, "utility"], name)
    verify_utility(geometry, utility_column="utility", name=name)
    leaked = geometry.groupby("episode_id")["meta_split"].nunique()
    leaked = leaked[leaked > 1]
    if not leaked.empty:
        raise ValueError(f"{name}: episode IDs appear in multiple splits: {list(leaked.index[:5])}.")
    for split, expected in {"train": 400, "validation": 100, "test": 100}.items():
        split_df = geometry[geometry["meta_split"] == split].copy()
        validate_four_config_grid(split_df, n_episodes=expected, name=f"{name} {split}")
        expected_per_objective = expected // len(OBJECTIVE_ORDER)
        counts = split_df.drop_duplicates("episode_id").groupby("learning_type")["episode_id"].count()
        bad = {obj: int(counts.get(obj, 0)) for obj in OBJECTIVE_ORDER if int(counts.get(obj, 0)) != expected_per_objective}
        if bad:
            raise ValueError(f"{name} {split}: expected {expected_per_objective} episodes/objective, got {bad}.")
    geometry.attrs["source_path"] = str(path)
    return geometry


def load_predictions(path: Path, *, name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    pred = read_csv(path)
    require_columns(
        pred,
        [
            "episode_id",
            "learning_type",
            "meta_split",
            "config_id",
            "observed_acquisition",
            "observed_transfer",
            "observed_boundedness",
            "observed_preservation",
            "observed_utility",
            "primary_predicted_utility",
        ],
        name,
    )
    if set(pred["meta_split"].astype(str).unique()) != {"test"}:
        raise ValueError(f"{name}: expected only TEST rows.")
    validate_four_config_grid(pred, n_episodes=100, name=name)
    ensure_numeric(
        pred,
        [
            "observed_acquisition",
            "observed_transfer",
            "observed_boundedness",
            "observed_preservation",
            "observed_utility",
            "primary_predicted_utility",
        ],
        name,
    )
    verify_utility(pred, utility_column="observed_utility", prefix="observed_", name=name)
    if "predicted_utility" in pred.columns:
        ensure_numeric(pred, ["predicted_utility"], name)
        diff = (pred["predicted_utility"] - pred["primary_predicted_utility"]).abs().max()
        if diff > 1e-9:
            raise ValueError(f"{name}: predicted_utility differs from primary_predicted_utility; max diff={diff}.")
    return pred


def choice_from_means(means: Mapping[str, float]) -> str:
    best = max(float(means[cfg]) for cfg in PROGRAM_ORDER)
    tied = [cfg for cfg in PROGRAM_ORDER if best - float(means[cfg]) <= TIE_TOLERANCE]
    # Match evaluate_compiler.py's deterministic max(..., config_id) tie break.
    return sorted(tied)[-1]


def oracle_by_episode(test: pd.DataFrame, *, utility_column: str = "utility") -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for episode_id, group in test.groupby("episode_id", sort=True):
        best = float(group[utility_column].max())
        oracle_set = sorted(group.loc[group[utility_column] >= best - TIE_TOLERANCE, "config_id"].astype(str))
        rows.append(
            {
                "episode_id": str(episode_id),
                "learning_type": str(group["learning_type"].iloc[0]),
                "oracle_utility": best,
                "oracle_config_set": "|".join(oracle_set),
                "oracle_set_size": len(oracle_set),
            }
        )
    return pd.DataFrame(rows)


def learn_global_default(geometry: pd.DataFrame) -> tuple[str, Dict[str, float]]:
    train = geometry[geometry["meta_split"] == "train"].copy()
    means = train.groupby("config_id")["utility"].mean().to_dict()
    if set(means) != set(PROGRAM_ORDER):
        raise ValueError("TRAIN geometry does not cover all candidate configs for global default.")
    return choice_from_means(means), {cfg: float(means[cfg]) for cfg in PROGRAM_ORDER}


def learn_objective_defaults(geometry: pd.DataFrame) -> tuple[Dict[str, str], Dict[str, Dict[str, float]]]:
    train = geometry[geometry["meta_split"] == "train"].copy()
    defaults: Dict[str, str] = {}
    means_by_objective: Dict[str, Dict[str, float]] = {}
    for objective in OBJECTIVE_ORDER:
        group = train[train["learning_type"] == objective]
        means = group.groupby("config_id")["utility"].mean().to_dict()
        if set(means) != set(PROGRAM_ORDER):
            raise ValueError(f"TRAIN geometry for {objective} does not cover all configs.")
        defaults[objective] = choice_from_means(means)
        means_by_objective[objective] = {cfg: float(means[cfg]) for cfg in PROGRAM_ORDER}
    return defaults, means_by_objective


def make_episode_row(
    *,
    panel: str,
    regime: str,
    method: str,
    episode_id: str,
    learning_type: str,
    selected_utility: float,
    oracle_utility: float,
    config_id: str,
    source_file: Path | str,
    policy_name: str,
) -> Dict[str, Any]:
    return {
        "panel": panel,
        "regime": regime,
        "method": method,
        "episode_id": episode_id,
        "learning_type": learning_type,
        "oracle_utility": float(oracle_utility),
        "selected_utility": float(selected_utility),
        "oracle_regret": float(oracle_utility) - float(selected_utility),
        "config_id": config_id,
        "source_file": str(source_file),
        "policy_name": policy_name,
    }


def selected_from_predictions(pred: pd.DataFrame, *, method: str, source_file: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for episode_id, group in pred.groupby("episode_id", sort=True):
        records = group.to_dict(orient="records")
        selected = max(records, key=lambda row: (float(row["primary_predicted_utility"]), str(row["config_id"])))
        oracle = float(group["observed_utility"].max())
        rows.append(
            make_episode_row(
                panel="A",
                regime="Experiment 4",
                method=method,
                episode_id=str(episode_id),
                learning_type=str(group["learning_type"].iloc[0]),
                selected_utility=float(selected["observed_utility"]),
                oracle_utility=oracle,
                config_id=str(selected["config_id"]),
                source_file=source_file,
                policy_name="compiler",
            )
        )
    return rows


def compiler_eval_rows(path: Path, *, panel: str, regime: str, method: str, expected_episodes: int) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    df = read_csv(path)
    require_columns(
        df,
        ["episode_id", "selected_config_id", "selected_observed_utility", "oracle_observed_utility"],
        str(path),
    )
    if len(df) != expected_episodes or df["episode_id"].nunique() != expected_episodes:
        raise ValueError(f"{path}: expected {expected_episodes} episode rows, found {len(df)}.")
    ensure_numeric(df, ["selected_observed_utility", "oracle_observed_utility"], str(path))
    if "oracle_regret" in df.columns:
        ensure_numeric(df, ["oracle_regret"], str(path))
    rows: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        regret = float(row["oracle_observed_utility"]) - float(row["selected_observed_utility"])
        if "oracle_regret" in df.columns and abs(float(row["oracle_regret"]) - regret) > 1e-8:
            raise ValueError(f"{path}: oracle_regret mismatch for {row['episode_id']}.")
        rows.append(
            make_episode_row(
                panel=panel,
                regime=regime,
                method=method,
                episode_id=str(row["episode_id"]),
                learning_type=str(row.get("learning_type", "")),
                selected_utility=float(row["selected_observed_utility"]),
                oracle_utility=float(row["oracle_observed_utility"]),
                config_id=str(row["selected_config_id"]),
                source_file=path,
                policy_name="compiler",
            )
        )
    return rows


def fixed_policy_rows(
    geometry: pd.DataFrame,
    *,
    panel: str,
    regime: str,
    method: str,
    fixed_config_by_episode: Mapping[str, str],
    source_file: Path,
    policy_name: str,
) -> List[Dict[str, Any]]:
    test = geometry[geometry["meta_split"] == "test"].copy()
    oracle = oracle_by_episode(test).set_index("episode_id")
    utilities = test.set_index(["episode_id", "config_id"])["utility"].astype(float).to_dict()
    learning_types = test.drop_duplicates("episode_id").set_index("episode_id")["learning_type"].astype(str).to_dict()
    rows: List[Dict[str, Any]] = []
    for episode_id in sorted(oracle.index):
        config_id = fixed_config_by_episode[str(episode_id)]
        selected = float(utilities[(str(episode_id), config_id)])
        rows.append(
            make_episode_row(
                panel=panel,
                regime=regime,
                method=method,
                episode_id=str(episode_id),
                learning_type=learning_types[str(episode_id)],
                selected_utility=selected,
                oracle_utility=float(oracle.loc[episode_id, "oracle_utility"]),
                config_id=config_id,
                source_file=source_file,
                policy_name=policy_name,
            )
        )
    return rows


def panel_a_rows(experiment2_root: Path) -> tuple[List[Dict[str, Any]], List[Path]]:
    rows: List[Dict[str, Any]] = []
    sources: List[Path] = []
    reference_grid: pd.DataFrame | None = None
    for rep_dir in REPRESENTATION_DIRS:
        method = REPRESENTATION_METHODS[rep_dir]
        release_root = experiment2_root / RELEASE_REPRESENTATION_PATHS[rep_dir]
        legacy_root = experiment2_root / rep_dir
        representation_root = release_root if release_root.exists() else legacy_root
        pred_path = representation_root / "predictions_test.csv"
        eval_path = representation_root / "compiler_evaluation.csv"
        pred = load_predictions(pred_path, name=f"Panel A {method}")
        grid = pred.sort_values(["episode_id", "config_id"])[
            ["episode_id", "learning_type", "config_id", "observed_utility"]
        ].reset_index(drop=True)
        if reference_grid is None:
            reference_grid = grid
        elif not grid.equals(reference_grid):
            raise ValueError(f"Panel A {method}: observed geometry grid differs from other representations.")
        if eval_path.exists():
            rep_rows = compiler_eval_rows(eval_path, panel="A", regime="Experiment 4", method=method, expected_episodes=100)
            sources.append(eval_path)
        else:
            rep_rows = selected_from_predictions(pred, method=method, source_file=pred_path)
            sources.append(pred_path)
        rows.extend(rep_rows)
    return rows, sources


def panel_b_rows(
    *,
    llama_geometry: pd.DataFrame,
    llama_geometry_path: Path,
    llama_compiler_path: Path,
    lofo_root: Path,
) -> tuple[List[Dict[str, Any]], str, List[Path]]:
    rows: List[Dict[str, Any]] = []
    sources = [llama_geometry_path, llama_compiler_path, lofo_root / "lofo_summary.csv"]
    global_default, _means = learn_global_default(llama_geometry)
    test_episodes = llama_geometry.loc[llama_geometry["meta_split"] == "test", "episode_id"].astype(str).unique()
    rows.extend(
        fixed_policy_rows(
            llama_geometry,
            panel="B",
            regime="Represented families",
            method="Global fixed",
            fixed_config_by_episode={ep: global_default for ep in test_episodes},
            source_file=llama_geometry_path,
            policy_name="global_fixed",
        )
    )
    rows.extend(
        compiler_eval_rows(
            llama_compiler_path,
            panel="B",
            regime="Represented families",
            method="Compiler",
            expected_episodes=100,
        )
    )

    lofo_episode_ids: set[str] = set()
    for objective in OBJECTIVE_ORDER:
        fold_path = lofo_root / objective / "compiler_selection.csv"
        metadata_path = lofo_root / objective / "model_metadata.json"
        if not fold_path.exists():
            raise FileNotFoundError(fold_path)
        if not metadata_path.exists():
            raise FileNotFoundError(metadata_path)
        sources.extend([fold_path, metadata_path])
        metadata = read_json(metadata_path)
        if metadata.get("held_out_learning_type") != objective:
            raise ValueError(f"{metadata_path}: held_out_learning_type mismatch.")
        if objective in set(metadata.get("training_learning_types", [])):
            raise ValueError(f"{metadata_path}: held-out objective appears in training types.")
        if objective in set(metadata.get("validation_learning_types", [])):
            raise ValueError(f"{metadata_path}: held-out objective appears in validation types.")
        if metadata.get("feature_set") != "episode":
            raise ValueError(f"{metadata_path}: expected episode-only LOFO feature_set.")
        fold = read_csv(fold_path)
        require_columns(
            fold,
            [
                "episode_id",
                "learning_type",
                "compiler_selected_config_id",
                "global_fixed_config_id",
                "compiler_observed_utility",
                "global_fixed_observed_utility",
                "oracle_observed_utility",
                "compiler_oracle_regret",
                "global_fixed_oracle_regret",
            ],
            str(fold_path),
        )
        if len(fold) != 20 or fold["episode_id"].nunique() != 20:
            raise ValueError(f"{fold_path}: expected 20 held-out test episodes, found rows={len(fold)}.")
        if set(fold["learning_type"].astype(str)) != {objective}:
            raise ValueError(f"{fold_path}: expected learning_type={objective}.")
        ensure_numeric(
            fold,
            [
                "compiler_observed_utility",
                "global_fixed_observed_utility",
                "oracle_observed_utility",
                "compiler_oracle_regret",
                "global_fixed_oracle_regret",
            ],
            str(fold_path),
        )
        for _, row in fold.iterrows():
            episode_id = str(row["episode_id"])
            if episode_id in lofo_episode_ids:
                raise ValueError(f"LOFO duplicate held-out episode ID: {episode_id}.")
            lofo_episode_ids.add(episode_id)
            compiler_regret = float(row["oracle_observed_utility"]) - float(row["compiler_observed_utility"])
            global_regret = float(row["oracle_observed_utility"]) - float(row["global_fixed_observed_utility"])
            if abs(compiler_regret - float(row["compiler_oracle_regret"])) > 1e-8:
                raise ValueError(f"{fold_path}: compiler regret mismatch for {episode_id}.")
            if abs(global_regret - float(row["global_fixed_oracle_regret"])) > 1e-8:
                raise ValueError(f"{fold_path}: global regret mismatch for {episode_id}.")
            rows.append(
                make_episode_row(
                    panel="B",
                    regime="Held-out family (LOFO)",
                    method="Global fixed",
                    episode_id=episode_id,
                    learning_type=objective,
                    selected_utility=float(row["global_fixed_observed_utility"]),
                    oracle_utility=float(row["oracle_observed_utility"]),
                    config_id=str(row["global_fixed_config_id"]),
                    source_file=fold_path,
                    policy_name="global_fixed",
                )
            )
            rows.append(
                make_episode_row(
                    panel="B",
                    regime="Held-out family (LOFO)",
                    method="Compiler",
                    episode_id=episode_id,
                    learning_type=objective,
                    selected_utility=float(row["compiler_observed_utility"]),
                    oracle_utility=float(row["oracle_observed_utility"]),
                    config_id=str(row["compiler_selected_config_id"]),
                    source_file=fold_path,
                    policy_name="compiler",
                )
            )
    if len(lofo_episode_ids) != 100:
        raise ValueError(f"LOFO expected 100 pooled held-out episodes, found {len(lofo_episode_ids)}.")
    return rows, global_default, sources


def panel_c_rows(
    *,
    llama_geometry: pd.DataFrame,
    llama_geometry_path: Path,
    llama_compiler_path: Path,
    gemma_geometry: pd.DataFrame,
    gemma_geometry_path: Path,
    gemma_compiler_path: Path,
) -> tuple[List[Dict[str, Any]], Dict[str, str], Dict[str, str], List[Path]]:
    rows: List[Dict[str, Any]] = []
    sources = [llama_geometry_path, llama_compiler_path, gemma_geometry_path, gemma_compiler_path]
    defaults_by_model: Dict[str, Dict[str, str]] = {}
    for regime, geometry, geometry_path, compiler_path in [
        ("Llama-3.1-8B", llama_geometry, llama_geometry_path, llama_compiler_path),
        ("Gemma-2-9B", gemma_geometry, gemma_geometry_path, gemma_compiler_path),
    ]:
        defaults, _means = learn_objective_defaults(geometry)
        defaults_by_model[regime] = defaults
        fixed = {
            str(row["episode_id"]): defaults[str(row["learning_type"])]
            for _, row in geometry[geometry["meta_split"] == "test"].drop_duplicates("episode_id").iterrows()
        }
        rows.extend(
            fixed_policy_rows(
                geometry,
                panel="C",
                regime=regime,
                method="Objective-fixed",
                fixed_config_by_episode=fixed,
                source_file=geometry_path,
                policy_name="objective_fixed",
            )
        )
        rows.extend(
            compiler_eval_rows(
                compiler_path,
                panel="C",
                regime=regime,
                method="Compiler",
                expected_episodes=100,
            )
        )
    return rows, defaults_by_model["Llama-3.1-8B"], defaults_by_model["Gemma-2-9B"], sources


def summarize_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[tuple[str, str, str], List[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row["panel"]), str(row["regime"]), str(row["method"])), []).append(row)
    out: List[Dict[str, Any]] = []
    for (panel, regime, method), group in sorted(grouped.items()):
        values = np.asarray([float(row["oracle_regret"]) for row in group], dtype=float)
        if values.size > 1 and panel != "A":
            mean, lo, hi = bootstrap_ci(values, n_boot=N_BOOT, seed=SEED)
        else:
            mean = float(values.mean())
            lo = float("nan")
            hi = float("nan")
        out.append(
            {
                "panel": panel,
                "regime": regime,
                "method": method,
                "value": mean,
                "ci_low": lo,
                "ci_high": hi,
                "n_episodes": int(values.size),
                "source_file": ";".join(sorted({str(row["source_file"]) for row in group})),
            }
        )
    return out


def assert_expected_aggregates(values: Sequence[Mapping[str, Any]]) -> None:
    expected = {
        ("A", "Experiment 4", "Episode"): 100,
        ("A", "Experiment 4", "Full"): 100,
        ("A", "Experiment 4", "Module probes"): 100,
        ("A", "Experiment 4", "Frozen behavior"): 100,
        ("B", "Represented families", "Global fixed"): 100,
        ("B", "Represented families", "Compiler"): 100,
        ("B", "Held-out family (LOFO)", "Global fixed"): 100,
        ("B", "Held-out family (LOFO)", "Compiler"): 100,
        ("C", "Llama-3.1-8B", "Objective-fixed"): 100,
        ("C", "Llama-3.1-8B", "Compiler"): 100,
        ("C", "Gemma-2-9B", "Objective-fixed"): 100,
        ("C", "Gemma-2-9B", "Compiler"): 100,
    }
    observed = {(row["panel"], row["regime"], row["method"]): int(row["n_episodes"]) for row in values}
    missing = sorted(set(expected) - set(observed))
    if missing:
        raise ValueError(f"Missing aggregate rows: {missing}")
    bad = {key: (observed[key], expected[key]) for key in expected if observed[key] != expected[key]}
    if bad:
        raise ValueError(f"Unexpected aggregate episode counts: {bad}")


def value_lookup(values: Sequence[Mapping[str, Any]]) -> Dict[tuple[str, str, str], float]:
    return {(str(row["panel"]), str(row["regime"]), str(row["method"])): float(row["value"]) for row in values}


def compare_summary(discrepancies: List[str], label: str, observed: float, expected: Any, tol: float = 1e-6) -> None:
    if expected in (None, ""):
        discrepancies.append(f"{label}: missing summary value")
        return
    diff = abs(float(observed) - float(expected))
    if diff > tol:
        discrepancies.append(f"{label}: derived={observed:.12g}, summary={float(expected):.12g}, diff={diff:.3g}")


def compare_current_summaries(
    values: Sequence[Mapping[str, Any]],
    *,
    llama_compiler_summary: Path,
    gemma_compiler_summary: Path,
    lofo_summary: Path,
) -> List[str]:
    lookup = value_lookup(values)
    discrepancies: List[str] = []
    if llama_compiler_summary.exists():
        summary = read_json(llama_compiler_summary)
        compare_summary(discrepancies, "Llama compiler Experiment 3 regret", lookup[("C", "Llama-3.1-8B", "Compiler")], summary.get("mean_oracle_regret"))
        compare_summary(discrepancies, "Llama represented compiler regret", lookup[("B", "Represented families", "Compiler")], summary.get("mean_oracle_regret"))
        compare_summary(discrepancies, "Panel A Full regret", lookup[("A", "Experiment 4", "Full")], summary.get("mean_oracle_regret"))
    if gemma_compiler_summary.exists():
        summary = read_json(gemma_compiler_summary)
        compare_summary(discrepancies, "Gemma compiler regret", lookup[("C", "Gemma-2-9B", "Compiler")], summary.get("mean_oracle_regret"))
    if lofo_summary.exists():
        lofo = read_csv(lofo_summary)
        macro = lofo[lofo["held_out_learning_type"] == "macro_average"]
        if macro.empty:
            discrepancies.append(f"{lofo_summary}: macro_average row missing")
        else:
            row = macro.iloc[0]
            compare_summary(discrepancies, "LOFO compiler regret", lookup[("B", "Held-out family (LOFO)", "Compiler")], row.get("compiler_oracle_regret"))
            compare_summary(discrepancies, "LOFO global fixed regret", lookup[("B", "Held-out family (LOFO)", "Global fixed")], row.get("global_fixed_oracle_regret"))
    return discrepancies


def draw_figure(values: Sequence[Mapping[str, Any]], output_dir: Path, source_paths: Sequence[Path]) -> tuple[Path, Path, Path]:
    lookup = {(row["panel"], row["regime"], row["method"]): row for row in values}
    plt = setup_matplotlib()
    fig, (ax_a, ax_b, ax_c) = plt.subplots(
        1,
        3,
        figsize=(7.95, 2.7),
        gridspec_kw={"width_ratios": [1.15, 1.05, 1.1]},
    )

    # Panel A
    methods_a = ["Episode", "Full", "Module probes", "Frozen behavior"]
    x = np.arange(len(methods_a))
    y = [lookup[("A", "Experiment 4", method)]["value"] for method in methods_a]
    bars = ax_a.bar(x, y, color=NEUTRAL_COLOR, alpha=0.78, width=0.62)
    ymax_a = max(y) if y else 0.02
    for i, (bar, method, val) in enumerate(zip(bars, methods_a, y)):
        if method == "Full":
            bar.set_edgecolor("#111111")
            bar.set_linewidth(1.5)
            ax_a.text(i, val + ymax_a * 0.13, "Primary\nrepresentation", ha="center", va="bottom", fontsize=6.8)
        ax_a.text(i, val + ymax_a * 0.03, f"{val:.3f}", ha="center", va="bottom", fontsize=7.2)
    ax_a.set_xticks(x, [REPRESENTATION_TICKS[m] for m in methods_a])
    ax_a.tick_params(axis="x", labelsize=7.4, pad=2)
    ax_a.set_ylabel("Oracle regret ↓")
    ax_a.set_title("A  Pre-adaptation information", loc="left", fontweight="bold", fontsize=9.3)
    ax_a.grid(axis="y")
    ax_a.set_axisbelow(True)
    ax_a.set_ylim(0, ymax_a * 1.5)

    # Panel B
    regimes_b = ["Represented families", "Held-out family (LOFO)"]
    methods_b = ["Global fixed", "Compiler"]
    labels_b = ["Represented\nfamilies", "Held-out family\n(LOFO)"]
    group_x = np.arange(len(regimes_b))
    width = 0.34
    colors_b = {"Global fixed": GLOBAL_COLOR, "Compiler": COMPILER_COLOR}
    offsets = {"Global fixed": -width / 2, "Compiler": width / 2}
    max_b = 0.0
    for method in methods_b:
        vals = [lookup[("B", regime, method)]["value"] for regime in regimes_b]
        lows = [lookup[("B", regime, method)]["ci_low"] for regime in regimes_b]
        highs = [lookup[("B", regime, method)]["ci_high"] for regime in regimes_b]
        yerr = np.array([[v - lo for v, lo in zip(vals, lows)], [hi - v for v, hi in zip(vals, highs)]], dtype=float)
        max_b = max(max_b, max(highs))
        ax_b.bar(group_x + offsets[method], vals, yerr=yerr, width=width, color=colors_b[method], alpha=0.86, capsize=2.5, label=method)
        for gx, val in zip(group_x + offsets[method], vals):
            ax_b.text(gx, val + 0.004, f"{val:.3f}", ha="center", va="bottom", fontsize=7.2)
    ax_b.set_xticks(group_x, labels_b)
    ax_b.set_ylabel("Oracle regret ↓")
    ax_b.set_title("B  Coverage of learning regime", loc="left", fontweight="bold", fontsize=9.3)
    ax_b.grid(axis="y")
    ax_b.set_axisbelow(True)
    ax_b.legend(frameon=False, fontsize=7.2, loc="upper left")
    ax_b.set_ylim(0, max_b + 0.025)

    # Panel C
    regimes_c = ["Llama-3.1-8B", "Gemma-2-9B"]
    methods_c = ["Objective-fixed", "Compiler"]
    labels_c = ["Llama-3.1-8B", "Gemma-2-9B"]
    colors_c = {"Objective-fixed": OBJECTIVE_COLOR, "Compiler": COMPILER_COLOR}
    offsets_c = {"Objective-fixed": -width / 2, "Compiler": width / 2}
    group_x = np.arange(len(regimes_c))
    max_c = 0.0
    for method in methods_c:
        vals = [lookup[("C", regime, method)]["value"] for regime in regimes_c]
        lows = [lookup[("C", regime, method)]["ci_low"] for regime in regimes_c]
        highs = [lookup[("C", regime, method)]["ci_high"] for regime in regimes_c]
        yerr = np.array([[v - lo for v, lo in zip(vals, lows)], [hi - v for v, hi in zip(vals, highs)]], dtype=float)
        max_c = max(max_c, max(highs))
        ax_c.bar(group_x + offsets_c[method], vals, yerr=yerr, width=width, color=colors_c[method], alpha=0.86, capsize=2.5, label=method)
        for gx, val in zip(group_x + offsets_c[method], vals):
            ax_c.text(gx, val + 0.0035, f"{val:.3f}", ha="center", va="bottom", fontsize=7.2)
    ax_c.set_xticks(group_x, labels_c)
    ax_c.set_ylabel("Oracle regret ↓")
    ax_c.set_title("C  Backbone dependence", loc="left", fontweight="bold", fontsize=9.3)
    ax_c.grid(axis="y")
    ax_c.set_axisbelow(True)
    ax_c.legend(frameon=False, fontsize=7.2, loc="upper left")
    ax_c.set_ylim(0, max_c + 0.02)

    for ax in [ax_a, ax_b, ax_c]:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout(w_pad=1.3)
    metadata = json_script_metadata(__file__, source_paths)
    metadata.update({"tie_tolerance": TIE_TOLERANCE, "bootstrap_resamples": N_BOOT})
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = save_outputs(fig, output_dir / "compilation_scope", metadata)
    plt.close(fig)
    return paths


def print_summary(
    values: Sequence[Mapping[str, Any]],
    *,
    llama_global_default: str,
    llama_objective_defaults: Mapping[str, str],
    gemma_objective_defaults: Mapping[str, str],
    discrepancies: Sequence[str],
    source_paths: Sequence[Path],
) -> None:
    lookup = {(row["panel"], row["regime"], row["method"]): row for row in values}
    print("\nPANEL A")
    for method in ["Episode", "Full", "Module probes", "Frozen behavior"]:
        row = lookup[("A", "Experiment 4", method)]
        print(f"- {method} regret: {format_number(row['value'])} (n={row['n_episodes']})")
    print("\nPANEL B")
    for regime, method in [
        ("Represented families", "Global fixed"),
        ("Represented families", "Compiler"),
        ("Held-out family (LOFO)", "Global fixed"),
        ("Held-out family (LOFO)", "Compiler"),
    ]:
        row = lookup[("B", regime, method)]
        print(f"- {regime} {method}: {format_number(row['value'])} (n={row['n_episodes']})")
    print("\nPANEL C")
    for regime, method in [
        ("Llama-3.1-8B", "Objective-fixed"),
        ("Llama-3.1-8B", "Compiler"),
        ("Gemma-2-9B", "Objective-fixed"),
        ("Gemma-2-9B", "Compiler"),
    ]:
        row = lookup[("C", regime, method)]
        print(f"- {regime} {method}: {format_number(row['value'])} (n={row['n_episodes']})")
    print("\nDefaults")
    print(f"- Llama global training-selected config: {llama_global_default}")
    print(f"- Llama objective-specific defaults: {dict(llama_objective_defaults)}")
    print(f"- Gemma objective-specific defaults: {dict(gemma_objective_defaults)}")
    print("\nSource paths used")
    for path in source_paths:
        print(f"- {path}")
    print("\nDiscrepancies > 1e-6")
    if discrepancies:
        for item in discrepancies:
            print(f"- {item}")
    else:
        print("- none")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot the compilation scope synthesis figure.")
    parser.add_argument("--llama-geometry", default="artifacts/llama/geometry/geometry_dataset.csv")
    parser.add_argument("--llama-experiment2-root", default="artifacts/llama")
    parser.add_argument("--llama-compiler-eval", default="artifacts/llama/selection/compiler_evaluation.csv")
    parser.add_argument("--llama-lofo-root", default="artifacts/llama/lofo")
    parser.add_argument("--gemma-geometry", default="artifacts/gemma/geometry/geometry_dataset.csv")
    parser.add_argument("--gemma-compiler-eval", default="artifacts/gemma/selection/compiler_evaluation.csv")
    parser.add_argument("--output-dir", default="outputs/release_verification/figures/synthesis")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    llama_geometry_path = Path(args.llama_geometry)
    llama_experiment2_root = Path(args.llama_experiment2_root)
    llama_compiler_path = Path(args.llama_compiler_eval)
    llama_lofo_root = Path(args.llama_lofo_root)
    gemma_geometry_path = Path(args.gemma_geometry)
    gemma_compiler_path = Path(args.gemma_compiler_eval)
    output_dir = Path(args.output_dir)

    llama_geometry = load_geometry(llama_geometry_path, name="current Llama geometry")
    gemma_geometry = load_geometry(gemma_geometry_path, name="corrected Gemma geometry")

    rows_a, sources_a = panel_a_rows(llama_experiment2_root)
    rows_b, llama_global_default, sources_b = panel_b_rows(
        llama_geometry=llama_geometry,
        llama_geometry_path=llama_geometry_path,
        llama_compiler_path=llama_compiler_path,
        lofo_root=llama_lofo_root,
    )
    rows_c, llama_objective_defaults, gemma_objective_defaults, sources_c = panel_c_rows(
        llama_geometry=llama_geometry,
        llama_geometry_path=llama_geometry_path,
        llama_compiler_path=llama_compiler_path,
        gemma_geometry=gemma_geometry,
        gemma_geometry_path=gemma_geometry_path,
        gemma_compiler_path=gemma_compiler_path,
    )
    episode_rows = [*rows_a, *rows_b, *rows_c]
    values = summarize_rows(episode_rows)
    assert_expected_aggregates(values)

    discrepancies = compare_current_summaries(
        values,
        llama_compiler_summary=llama_compiler_path.with_name("compiler_evaluation_summary.json"),
        gemma_compiler_summary=gemma_compiler_path.with_name("compiler_evaluation_summary.json"),
        lofo_summary=llama_lofo_root / "lofo_summary.csv",
    )
    source_paths = list(dict.fromkeys([*sources_a, *sources_b, *sources_c]))
    figure_paths = draw_figure(values, output_dir, source_paths)
    values_path = write_csv(output_dir / "compilation_scope_values.csv", values)
    episode_path = write_csv(output_dir / "compilation_scope_episode_values.csv", episode_rows)
    print_summary(
        values,
        llama_global_default=llama_global_default,
        llama_objective_defaults=llama_objective_defaults,
        gemma_objective_defaults=gemma_objective_defaults,
        discrepancies=discrepancies,
        source_paths=source_paths,
    )
    print("\nGenerated")
    for path in [*figure_paths, values_path, episode_path]:
        print(f"- {path}")


if __name__ == "__main__":
    main()
