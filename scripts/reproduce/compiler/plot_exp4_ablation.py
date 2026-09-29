#!/usr/bin/env python3
"""Main Experiment 4: representation ablation figure.

This script only reads saved Experiment 2 ablation outputs. It does not rerun
feature extraction, predictor training, or adaptation jobs.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from plot_style import (
    FEATURE_SET_LABELS,
    PROGRAM_ORDER,
    SELECTOR_COLORS,
    ensure_dirs,
    format_number,
    json_script_metadata,
    panel_label,
    prediction_selected_utilities,
    prettify_axes,
    read_csv,
    read_json,
    save_outputs,
    setup_matplotlib,
    write_summary,
)


FEATURE_DIRS = [
    "ablation_episode",
    "full_auto",
    "ablation_module_probes",
    "ablation_frozen_behavior",
]
FEATURE_PATHS = {
    "ablation_episode": Path("ablations/episode"),
    "full_auto": Path("prediction"),
    "ablation_module_probes": Path("ablations/module_probes"),
    "ablation_frozen_behavior": Path("ablations/frozen_behavior"),
}
EXPECTED_FEATURE_SETS = {
    "ablation_episode": "episode",
    "full_auto": "full",
    "ablation_module_probes": "module_probes",
    "ablation_frozen_behavior": "frozen_behavior",
}
ROW_LABELS = {
    "ablation_episode": "Episode",
    "full_auto": "Full (primary)",
    "ablation_module_probes": "Module probes",
    "ablation_frozen_behavior": "Frozen behavior",
}
EXPECTED_SANITY = {
    "ablation_episode": {
        "mean_mae": 0.0398721,
        "top1_oracle_recovery": 0.83,
        "compiler_oracle_regret": 0.00883,
    },
    "full_auto": {
        "mean_mae": 0.0400217,
        "top1_oracle_recovery": 0.77,
        "compiler_oracle_regret": 0.0112604,
    },
    "ablation_module_probes": {
        "mean_mae": 0.0530148,
        "top1_oracle_recovery": 0.74,
        "compiler_oracle_regret": 0.01407,
    },
    "ablation_frozen_behavior": {
        "mean_mae": 0.0538471,
        "top1_oracle_recovery": 0.71,
        "compiler_oracle_regret": 0.01858,
    },
}
GEOMETRY_COLUMNS = [
    "model_slug",
    "episode_id",
    "learning_type",
    "meta_split",
    "config_id",
    "number_of_seeds",
    "seed_values",
    "observed_acquisition",
    "observed_transfer",
    "observed_boundedness",
    "observed_preservation",
    "observed_utility",
]


def _same_selected_hyperparameters(metadata: Mapping[str, Any]) -> bool:
    selected = dict(metadata.get("selected_hyperparameters") or {})
    candidates = metadata.get("validation_model_selection_results") or []
    if not selected or not candidates:
        return False
    for candidate in candidates:
        if all(candidate.get(key) == value for key, value in selected.items()):
            return True
    return False


def _assert_close(name: str, value: float, expected: float, tolerance: float) -> None:
    if abs(float(value) - float(expected)) > tolerance:
        raise ValueError(
            f"{name} changed materially: observed {value:.8f}, expected about {expected:.8f} "
            f"(tolerance {tolerance})."
        )


def _load_one(root: Path, feature_dir: str) -> Dict[str, Any]:
    release_root = root / FEATURE_PATHS[feature_dir]
    legacy_root = root / feature_dir
    run_root = release_root if release_root.exists() else legacy_root
    metrics_path = run_root / "prediction_metrics.json"
    ranking_path = run_root / "ranking_metrics.json"
    predictions_path = run_root / "predictions_test.csv"
    metadata_path = run_root / "model_metadata.json"
    missing = [p for p in [metrics_path, ranking_path, predictions_path, metadata_path] if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing ablation source files: {missing}")

    predictions = read_csv(predictions_path)
    metrics = read_json(metrics_path)
    ranking = read_json(ranking_path)
    metadata = read_json(metadata_path)
    selected = prediction_selected_utilities(predictions, method_prefix="primary")
    return {
        "feature_dir": feature_dir,
        "run_root": run_root,
        "metrics_path": metrics_path,
        "ranking_path": ranking_path,
        "predictions_path": predictions_path,
        "metadata_path": metadata_path,
        "metrics": metrics,
        "ranking": ranking,
        "predictions": predictions,
        "metadata": metadata,
        "selected": selected,
    }


def validate_sources(loaded: Sequence[Mapping[str, Any]], sanity_tolerance: float) -> Dict[str, Any]:
    checks: Dict[str, Any] = {}
    if len(loaded) != len(FEATURE_DIRS):
        raise ValueError(f"Expected {len(FEATURE_DIRS)} representation results, found {len(loaded)}.")

    reference = loaded[0]["predictions"].sort_values(["episode_id", "config_id"]).reset_index(drop=True)
    reference_geometry = reference[GEOMETRY_COLUMNS]
    reference_episodes = set(reference["episode_id"].astype(str))
    reference_configs = set(reference["config_id"].astype(str))

    if reference["episode_id"].nunique() != 100:
        raise ValueError(f"Expected 100 held-out test episodes, found {reference['episode_id'].nunique()}.")
    if len(reference) != 100 * len(PROGRAM_ORDER):
        raise ValueError(f"Expected 400 test rows, found {len(reference)}.")
    if reference_configs != set(PROGRAM_ORDER):
        raise ValueError(f"Expected config IDs {PROGRAM_ORDER}, found {sorted(reference_configs)}.")

    for item in loaded:
        feature_dir = str(item["feature_dir"])
        metadata = item["metadata"]
        predictions = item["predictions"].sort_values(["episode_id", "config_id"]).reset_index(drop=True)

        if set(predictions["meta_split"].astype(str)) != {"test"}:
            raise ValueError(f"{feature_dir} predictions_test.csv contains non-test rows.")
        if predictions["episode_id"].nunique() != 100 or len(predictions) != 100 * len(PROGRAM_ORDER):
            raise ValueError(f"{feature_dir} does not contain the same 100 episode x 4 config grid.")
        if set(predictions["episode_id"].astype(str)) != reference_episodes:
            raise ValueError(f"{feature_dir} uses a different held-out episode set.")
        if set(predictions["config_id"].astype(str)) != reference_configs:
            raise ValueError(f"{feature_dir} uses a different candidate-program set.")

        geometry = predictions[GEOMETRY_COLUMNS]
        non_float_cols = [c for c in GEOMETRY_COLUMNS if not c.startswith("observed_")]
        if not geometry[non_float_cols].astype(str).equals(reference_geometry[non_float_cols].astype(str)):
            raise ValueError(f"{feature_dir} does not share the same episode/config metadata grid.")
        for column in [c for c in GEOMETRY_COLUMNS if c.startswith("observed_")]:
            if not np.allclose(geometry[column].astype(float), reference_geometry[column].astype(float), atol=1e-12):
                raise ValueError(f"{feature_dir} does not share the same seed-averaged observed geometry.")

        expected_feature_set = EXPECTED_FEATURE_SETS[feature_dir]
        if metadata.get("feature_set") != expected_feature_set:
            raise ValueError(
                f"{feature_dir} metadata feature_set={metadata.get('feature_set')!r}; "
                f"expected {expected_feature_set!r}."
            )
        if bool(metadata.get("objective_identity_used_in_primary_predictor")):
            raise ValueError(f"{feature_dir} primary predictor unexpectedly used objective identity.")
        if not bool(metadata.get("objective_conditioned_mean_is_diagnostic_only")):
            raise ValueError(f"{feature_dir} does not mark objective-conditioned mean as diagnostic only.")
        if metadata.get("config_ids") != PROGRAM_ORDER:
            raise ValueError(f"{feature_dir} metadata config_ids do not match the primary four-program library.")
        if int(metadata.get("train_episodes", -1)) <= 0 or int(metadata.get("validation_episodes", -1)) <= 0:
            raise ValueError(f"{feature_dir} metadata is missing train/validation episode counts.")
        if not _same_selected_hyperparameters(metadata):
            raise ValueError(f"{feature_dir} selected hyperparameters are not found in validation-selection results.")

    full_metadata = next(item["metadata"] for item in loaded if item["feature_dir"] == "full_auto")
    if full_metadata.get("feature_set") != "full":
        raise ValueError("full_auto is not recorded as the full primary representation.")

    checks["same_held_out_test_episodes"] = True
    checks["same_candidate_programs"] = list(PROGRAM_ORDER)
    checks["same_seed_averaged_test_geometry"] = True
    checks["validation_selection_metadata_present"] = True
    checks["full_auto_is_primary_representation"] = True

    rows = collect_rows_from_loaded(loaded)
    by_dir = {row["feature_dir"]: row for row in rows}
    for feature_dir, expected in EXPECTED_SANITY.items():
        row = by_dir[feature_dir]
        for metric, expected_value in expected.items():
            _assert_close(f"{feature_dir} {metric}", row[metric], expected_value, sanity_tolerance)
    checks["sanity_values_within_tolerance"] = sanity_tolerance
    return checks


def collect_rows_from_loaded(loaded: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for item in loaded:
        feature_dir = str(item["feature_dir"])
        metrics = item["metrics"]
        ranking = item["ranking"]
        selected = item["selected"]
        predictions = item["predictions"]
        row = {
            "feature_dir": feature_dir,
            "representation": ROW_LABELS[feature_dir].replace(" (primary)", ""),
            "representation_plot_label": ROW_LABELS[feature_dir],
            "feature_set_label": FEATURE_SET_LABELS[feature_dir].replace("\n", " "),
            "metadata_feature_set": item["metadata"].get("feature_set"),
            "mean_mae": float(metrics["primary"]["mean_mae"]),
            "utility_correlation": float(metrics["utility_correlation"]["primary"]),
            "mean_episode_spearman": float(ranking["primary"]["mean_episode_spearman"]),
            "pairwise_ranking_accuracy": float(ranking["primary"]["pairwise_ranking_accuracy"]),
            "top1_oracle_recovery": float(ranking["primary"]["top1_set_oracle_recovery"]),
            "top2_oracle_recovery": float(ranking["primary"]["top2_set_oracle_recovery"]),
            "selected_mean_utility": float(selected["selected_observed_utility"].mean()),
            "oracle_mean_utility": float(selected["oracle_observed_utility"].mean()),
            "compiler_oracle_regret": float(selected["oracle_regret"].mean()),
            "n_test_episodes": int(predictions["episode_id"].nunique()),
            "test_rows": int(len(predictions)),
            "config_ids": "|".join(PROGRAM_ORDER),
            "prediction_metrics_json": str(item["metrics_path"]),
            "ranking_metrics_json": str(item["ranking_path"]),
            "predictions_test_csv": str(item["predictions_path"]),
            "model_metadata_json": str(item["metadata_path"]),
        }
        rows.append(row)
    return rows


def collect_rows(experiment2_root: Path, sanity_tolerance: float) -> Tuple[List[Dict[str, Any]], List[Path], Dict[str, Any]]:
    loaded = [_load_one(experiment2_root, feature_dir) for feature_dir in FEATURE_DIRS]
    checks = validate_sources(loaded, sanity_tolerance=sanity_tolerance)
    rows = collect_rows_from_loaded(loaded)
    inputs: List[Path] = []
    for item in loaded:
        inputs.extend([item["metrics_path"], item["ranking_path"], item["predictions_path"], item["metadata_path"]])
    return rows, inputs, checks


def _point(ax: Any, x: float, y: float, is_primary: bool) -> None:
    if is_primary:
        ax.scatter(
            [x],
            [y],
            s=48,
            marker="o",
            facecolor="white",
            edgecolor=SELECTOR_COLORS["primary"],
            linewidth=1.4,
            zorder=3,
        )
    else:
        ax.scatter(
            [x],
            [y],
            s=46,
            marker="o",
            facecolor=SELECTOR_COLORS["primary"],
            edgecolor="white",
            linewidth=0.6,
            zorder=3,
        )


def draw_figure(rows: Sequence[Mapping[str, Any]], output_root: Path, inputs: Sequence[Path], checks: Mapping[str, Any]) -> Dict[str, Any]:
    plt = setup_matplotlib()
    from matplotlib.ticker import PercentFormatter

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.65), sharey=True)
    y = np.arange(len(rows))
    labels = [str(row["representation_plot_label"]) for row in rows]
    primary_color = SELECTOR_COLORS["primary"]
    label_color = "#303030"

    panels = [
        {
            "ax": axes[0],
            "title": "Geometry prediction",
            "xlabel": "Mean absolute error",
            "metric": "mean_mae",
            "xlim": (0.0, 0.068),
            "label": lambda value: format_number(value, 3),
            "offset": 0.0022,
            "panel": "A",
        },
        {
            "ax": axes[1],
            "title": "Selection quality",
            "xlabel": "Oracle regret",
            "metric": "compiler_oracle_regret",
            "xlim": (0.0, 0.024),
            "label": lambda value: format_number(value, 3),
            "offset": 0.0008,
            "panel": "B",
        },
        {
            "ax": axes[2],
            "title": "Oracle recovery",
            "xlabel": "Top-1 recovery",
            "metric": "top1_oracle_recovery",
            "xlim": (0.0, 1.0),
            "label": lambda value: f"{100 * float(value):.0f}%",
            "offset": 0.035,
            "panel": "C",
        },
    ]

    for panel in panels:
        ax = panel["ax"]
        ax.set_title(panel["title"], loc="left", pad=7)
        ax.set_xlabel(panel["xlabel"])
        ax.set_xlim(*panel["xlim"])
        ax.set_ylim(len(rows) - 0.5, -0.5)
        ax.set_yticks(y)
        ax.tick_params(axis="y", length=0)
        if panel["metric"] == "top1_oracle_recovery":
            ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
            ax.set_xticks([0.0, 0.5, 1.0])
        for i, row in enumerate(rows):
            value = float(row[panel["metric"]])
            is_primary = row["feature_dir"] == "full_auto"
            _point(ax, value, float(i), is_primary=is_primary)
            ax.text(
                value + float(panel["offset"]),
                i,
                panel["label"](value),
                ha="left",
                va="center",
                fontsize=7.7,
                color=label_color,
                clip_on=False,
            )
        prettify_axes(ax)
        panel_label(ax, panel["panel"])

    axes[0].set_yticklabels(labels)
    axes[1].tick_params(axis="y", labelleft=False)
    axes[2].tick_params(axis="y", labelleft=False)
    for ax in axes:
        ax.spines["left"].set_visible(False)
        ax.grid(axis="y", color="#E5E5E5", linewidth=0.7)
    axes[0].tick_params(axis="y", pad=4)
    fig.tight_layout(w_pad=2.0)

    dirs = ensure_dirs(output_root)
    summary_path = dirs["data"] / "exp4_ablation_summary.csv"
    write_summary(summary_path, rows)
    metadata = json_script_metadata(__file__, inputs)
    metadata.update(
        {
            "figure": "fig_exp4_ablation",
            "summary_csv": str(summary_path),
            "summary": rows,
            "data_integrity_checks": checks,
            "caption": (
                "Representation ablations. Geometry prediction and downstream selection are compared across "
                "the full episode--model representation and three restricted feature sets. (A) Mean "
                "adaptation-geometry prediction error. (B) Oracle regret of the program selected from "
                "predicted geometry. (C) Tie-aware top-1 recovery of an oracle-optimal program. The episode "
                "representation alone retains performance comparable to the full representation, while "
                "module-level probes and frozen-model behavioral statistics are weaker when used independently."
            ),
            "visual_note": (
                "All panels share the same representation rows. The outlined marker denotes the full primary "
                "representation selected before test evaluation."
            ),
        }
    )
    save_outputs(fig, dirs["main"] / "fig_exp4_ablation", metadata, dpi=300)
    plt.close(fig)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot main Experiment 4 representation-ablation figure.")
    parser.add_argument("--experiment2_root", default="artifacts/llama")
    parser.add_argument("--output_root", default="outputs/release_verification/figures/llama")
    parser.add_argument(
        "--sanity_tolerance",
        type=float,
        default=0.002,
        help="Maximum allowed absolute deviation from recorded ablation sanity values before plotting.",
    )
    args = parser.parse_args()

    rows, inputs, checks = collect_rows(Path(args.experiment2_root), sanity_tolerance=args.sanity_tolerance)
    draw_figure(rows, Path(args.output_root), inputs, checks)
    print("Experiment 4 ablation values:")
    for row in rows:
        print(
            "  "
            f"{row['representation']}: "
            f"MAE={row['mean_mae']:.6f}, "
            f"regret={row['compiler_oracle_regret']:.6f}, "
            f"top1={100 * row['top1_oracle_recovery']:.0f}%"
        )
    print(f"Wrote {Path(args.output_root) / 'main' / 'fig_exp4_ablation.pdf'}")
    print(f"Wrote {Path(args.output_root) / 'main' / 'fig_exp4_ablation.png'}")


if __name__ == "__main__":
    main()
