#!/usr/bin/env python3
"""Leave-one-objective-out Experiment 2 geometry generalization.

This entry point deliberately reuses the IID Experiment 2 feature encoding,
model-selection, ranking, and metric helpers while changing only the data
partitioning protocol.  For each held-out learning objective, the predictor is
trained and selected on the other objectives, then evaluated on held-out test
episodes only.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

try:  # pragma: no cover
    from .compiler_common import write_csv
    from .experiment2_geometry_predictor import (
        assert_complete_config_coverage,
        assert_feature_names_safe,
        assert_no_split_leakage,
        assert_seed_aggregated,
        build_prediction_rows,
        build_supervised_rows,
        config_mean_predictions,
        first_representation_metadata,
        fit_standardizer,
        git_commit,
        load_config_library,
        metric_mae,
        metric_rmse,
        pearson,
        predict_primary,
        prediction_metric_rows,
        prediction_metric_summary,
        rank_episode,
        read_csv_rows,
        read_feature_payloads,
        save_trained_model,
        selected_config_with_lexicographic_tie_break,
        summarize_ranking,
        top_k_set,
        train_primary_predictor,
        utility,
    )
    from .experiment2_geometry_predictor import apply_standardizer
    from .prepare_geometry_dataset import OUTCOME_COLUMNS, PRIMARY_CONFIGS
except ImportError:  # pragma: no cover
    from compiler_common import write_csv
    from experiment2_geometry_predictor import (
        assert_complete_config_coverage,
        assert_feature_names_safe,
        assert_no_split_leakage,
        assert_seed_aggregated,
        build_prediction_rows,
        build_supervised_rows,
        config_mean_predictions,
        first_representation_metadata,
        fit_standardizer,
        git_commit,
        load_config_library,
        metric_mae,
        metric_rmse,
        pearson,
        predict_primary,
        prediction_metric_rows,
        prediction_metric_summary,
        rank_episode,
        read_csv_rows,
        read_feature_payloads,
        save_trained_model,
        selected_config_with_lexicographic_tie_break,
        summarize_ranking,
        top_k_set,
        train_primary_predictor,
        utility,
    )
    from experiment2_geometry_predictor import apply_standardizer
    from prepare_geometry_dataset import OUTCOME_COLUMNS, PRIMARY_CONFIGS


LEARNING_TYPE_FORBIDDEN_FEATURE_TOKENS = ("learning_type", "objective", "objective_id")


def mean(values: Iterable[float | int | None]) -> float | None:
    vals = [float(v) for v in values if v is not None]
    return float(np.mean(vals)) if vals else None


def as_python(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {k: as_python(v) for k, v in value.items()}
    if isinstance(value, list):
        return [as_python(v) for v in value]
    return value


def write_json(path: str | Path, payload: Dict[str, Any] | List[Dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(as_python(payload), indent=2), encoding="utf-8")


def learning_type(row: Dict[str, Any]) -> str:
    record = row.get("record", row)
    return str(record["learning_type"])


def meta_split(row: Dict[str, Any]) -> str:
    record = row.get("record", row)
    return str(record["meta_split"])


def episode_id(row: Dict[str, Any]) -> str:
    record = row.get("record", row)
    return str(record["episode_id"])


def config_id(row: Dict[str, Any]) -> str:
    record = row.get("record", row)
    return str(record["config_id"])


def objective_list_from_geometry(rows: Sequence[Dict[str, Any]]) -> List[str]:
    objectives = sorted({str(row["learning_type"]) for row in rows})
    if not objectives:
        raise ValueError("Geometry dataset contains no learning_type values.")
    return objectives


def assert_objective_identity_absent(feature_names: Sequence[str]) -> None:
    assert_feature_names_safe(feature_names)
    for name in feature_names:
        lowered = name.lower()
        if any(token in lowered for token in LEARNING_TYPE_FORBIDDEN_FEATURE_TOKENS):
            raise ValueError(f"Objective identity leaked into primary feature matrix: {name}")


def assert_disjoint_episode_sets(
    train_rows: Sequence[Dict[str, Any]],
    val_rows: Sequence[Dict[str, Any]],
    test_rows: Sequence[Dict[str, Any]],
) -> None:
    train_ids = {episode_id(row) for row in train_rows}
    val_ids = {episode_id(row) for row in val_rows}
    test_ids = {episode_id(row) for row in test_rows}
    overlaps = {
        "train_validation": train_ids & val_ids,
        "train_test": train_ids & test_ids,
        "validation_test": val_ids & test_ids,
    }
    bad = {key: sorted(value)[:5] for key, value in overlaps.items() if value}
    if bad:
        raise ValueError(f"LOFO train/validation/test episode IDs overlap: {bad}")


def assert_complete_supervised_config_coverage(
    rows: Sequence[Dict[str, Any]],
    config_ids: Sequence[str],
    label: str,
) -> None:
    expected = set(config_ids)
    by_episode: Dict[str, set[str]] = defaultdict(set)
    for row in rows:
        by_episode[episode_id(row)].add(config_id(row))
    incomplete = {
        ep: sorted(expected - present)
        for ep, present in by_episode.items()
        if present != expected
    }
    if incomplete:
        first = next(iter(incomplete.items()))
        raise ValueError(f"{label} episode lacks complete four-config coverage: {first}")


def standard_dataset_expected(rows: Sequence[Dict[str, Any]], config_ids: Sequence[str]) -> bool:
    if len(config_ids) != 4:
        return False
    by_split: Dict[str, set[str]] = defaultdict(set)
    by_split_type: Dict[Tuple[str, str], set[str]] = defaultdict(set)
    for row in rows:
        by_split[str(row["meta_split"])].add(str(row["episode_id"]))
        by_split_type[(str(row["meta_split"]), str(row["learning_type"]))].add(str(row["episode_id"]))
    split_ok = {split: len(ids) for split, ids in by_split.items()} == {
        "train": 400,
        "validation": 100,
        "test": 100,
    }
    if not split_ok:
        return False
    objectives = {key[1] for key in by_split_type}
    if len(objectives) != 5:
        return False
    for objective in objectives:
        if len(by_split_type.get(("train", objective), set())) != 80:
            return False
        if len(by_split_type.get(("validation", objective), set())) != 20:
            return False
        if len(by_split_type.get(("test", objective), set())) != 20:
            return False
    return True


def lofo_split_rows(
    dataset: Sequence[Dict[str, Any]],
    held_out_learning_type: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    train_rows = [
        row for row in dataset
        if meta_split(row) == "train" and learning_type(row) != held_out_learning_type
    ]
    val_rows = [
        row for row in dataset
        if meta_split(row) == "validation" and learning_type(row) != held_out_learning_type
    ]
    test_rows = [
        row for row in dataset
        if meta_split(row) == "test" and learning_type(row) == held_out_learning_type
    ]
    return train_rows, val_rows, test_rows


def validate_lofo_partition(
    train_rows: Sequence[Dict[str, Any]],
    val_rows: Sequence[Dict[str, Any]],
    test_rows: Sequence[Dict[str, Any]],
    held_out_learning_type: str,
    config_ids: Sequence[str],
    strict_standard_counts: bool,
) -> Dict[str, Any]:
    if not train_rows:
        raise ValueError(f"No LOFO training rows for held_out_learning_type={held_out_learning_type}.")
    if not val_rows:
        raise ValueError(f"No LOFO validation rows for held_out_learning_type={held_out_learning_type}.")
    if not test_rows:
        raise ValueError(f"No LOFO test rows for held_out_learning_type={held_out_learning_type}.")

    train_types = {learning_type(row) for row in train_rows}
    val_types = {learning_type(row) for row in val_rows}
    test_types = {learning_type(row) for row in test_rows}
    if held_out_learning_type in train_types:
        raise ValueError("Held-out learning_type occurs in LOFO model-training rows.")
    if held_out_learning_type in val_types:
        raise ValueError("Held-out learning_type occurs in LOFO validation/model-selection rows.")
    if test_types != {held_out_learning_type}:
        raise ValueError(f"LOFO test rows must contain only {held_out_learning_type}; observed {sorted(test_types)}.")

    assert_disjoint_episode_sets(train_rows, val_rows, test_rows)
    assert_complete_supervised_config_coverage(test_rows, config_ids, "held-out test")
    assert_complete_supervised_config_coverage(train_rows, config_ids, "LOFO train")
    assert_complete_supervised_config_coverage(val_rows, config_ids, "LOFO validation")

    counts = {
        "train_rows": len(train_rows),
        "validation_rows": len(val_rows),
        "test_rows": len(test_rows),
        "train_episodes": len({episode_id(row) for row in train_rows}),
        "validation_episodes": len({episode_id(row) for row in val_rows}),
        "test_episodes": len({episode_id(row) for row in test_rows}),
        "training_learning_types": sorted(train_types),
        "validation_learning_types": sorted(val_types),
        "test_learning_types": sorted(test_types),
    }
    if strict_standard_counts:
        expected = {
            "train_rows": 1280,
            "validation_rows": 320,
            "test_rows": 80,
            "train_episodes": 320,
            "validation_episodes": 80,
            "test_episodes": 20,
        }
        bad = {key: (counts[key], value) for key, value in expected.items() if counts[key] != value}
        if bad:
            raise ValueError(f"Unexpected LOFO standard-dataset counts for {held_out_learning_type}: {bad}")
    return counts


def learn_global_fixed_program(
    train_rows: Sequence[Dict[str, Any]],
    config_ids: Sequence[str],
    tie_tolerance: float,
) -> Dict[str, Any]:
    utilities: Dict[str, List[float]] = defaultdict(list)
    for row in train_rows:
        utilities[config_id(row)].append(utility(row["y"]))
    missing = [cfg for cfg in config_ids if cfg not in utilities]
    if missing:
        raise ValueError(f"Cannot learn global-fixed baseline; missing train rows for {missing}")
    means = {cfg: float(np.mean(utilities[cfg])) for cfg in config_ids}
    best = max(means.values())
    tie_set = sorted(cfg for cfg, value in means.items() if best - value <= tie_tolerance)
    return {
        "selected_config_id": tie_set[0],
        "tie_set": tie_set,
        "train_mean_utility_by_config": means,
        "selection_rule": "highest mean balanced utility on LOFO training rows only; lexicographic tie break",
    }


def metric_predictions_summary(
    predictions_by_method: Dict[str, np.ndarray],
    gold: np.ndarray,
) -> Dict[str, Dict[str, Any]]:
    rows = prediction_metric_rows(predictions_by_method, gold)
    summary = prediction_metric_summary(rows)
    return summary


def build_ranking_rows(
    prediction_rows: Sequence[Dict[str, Any]],
    methods: Sequence[str],
    config_ids: Sequence[str],
    tie_tolerance: float,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    by_episode: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in prediction_rows:
        by_episode[str(row["episode_id"])].append(row)
    for ep, rows in sorted(by_episode.items()):
        if {row["config_id"] for row in rows} != set(config_ids):
            raise ValueError(f"LOFO held-out test episode lacks all configs: {ep}")
        ordered = sorted(rows, key=lambda row: str(row["config_id"]))
        for method in methods:
            out.append(rank_episode(ordered, method=method, tolerance=tie_tolerance))
    return out


def build_compiler_selection_rows(
    prediction_rows: Sequence[Dict[str, Any]],
    global_fixed: Dict[str, Any],
    config_ids: Sequence[str],
    tie_tolerance: float,
) -> List[Dict[str, Any]]:
    fixed_config = str(global_fixed["selected_config_id"])
    by_episode: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in prediction_rows:
        by_episode[str(row["episode_id"])].append(row)

    out: List[Dict[str, Any]] = []
    for ep, rows in sorted(by_episode.items()):
        if {row["config_id"] for row in rows} != set(config_ids):
            raise ValueError(f"Cannot score compiler selection; incomplete configs for {ep}")
        ordered = sorted(rows, key=lambda row: str(row["config_id"]))
        configs = [str(row["config_id"]) for row in ordered]
        observed = [float(row["observed_utility"]) for row in ordered]
        predicted = [float(row["primary_predicted_utility"]) for row in ordered]
        observed_by_config = dict(zip(configs, observed))
        best_observed = max(observed)
        oracle_set = sorted(cfg for cfg, value in zip(configs, observed) if best_observed - value <= tie_tolerance)
        compiler_config = selected_config_with_lexicographic_tie_break(configs, predicted, tie_tolerance)
        top1_set = top_k_set(configs, predicted, 1, tie_tolerance)
        top2_set = top_k_set(configs, predicted, 2, tie_tolerance)
        compiler_utility = observed_by_config[compiler_config]
        fixed_utility = observed_by_config[fixed_config]

        out.append(
            {
                "episode_id": ep,
                "learning_type": ordered[0]["learning_type"],
                "compiler_selected_config_id": compiler_config,
                "compiler_predicted_top1_set": "|".join(top1_set),
                "compiler_predicted_top2_set": "|".join(top2_set),
                "global_fixed_config_id": fixed_config,
                "oracle_config_ids": "|".join(oracle_set),
                "oracle_set_size": len(oracle_set),
                "compiler_observed_utility": compiler_utility,
                "global_fixed_observed_utility": fixed_utility,
                "oracle_observed_utility": best_observed,
                "compiler_oracle_regret": best_observed - compiler_utility,
                "global_fixed_oracle_regret": best_observed - fixed_utility,
                "compiler_oracle_optimal": compiler_config in set(oracle_set),
                "global_fixed_oracle_optimal": fixed_config in set(oracle_set),
            }
        )
    return out


def summarize_compiler_selection(
    rows: Sequence[Dict[str, Any]],
    global_fixed: Dict[str, Any],
) -> Dict[str, Any]:
    if not rows:
        raise ValueError("Cannot summarize an empty compiler-selection table.")
    selected_counts = Counter(str(row["compiler_selected_config_id"]) for row in rows)
    oracle_membership = Counter()
    for row in rows:
        for cfg in str(row["oracle_config_ids"]).split("|"):
            if cfg:
                oracle_membership[cfg] += 1
    return {
        "n_test_episodes": len(rows),
        "selected_config_distribution": dict(sorted(selected_counts.items())),
        "oracle_config_distribution": dict(sorted(oracle_membership.items())),
        "global_fixed_config_id": global_fixed["selected_config_id"],
        "global_fixed_tie_set": global_fixed["tie_set"],
        "global_fixed_train_mean_utility_by_config": global_fixed["train_mean_utility_by_config"],
        "compiler_mean_utility": mean(row["compiler_observed_utility"] for row in rows),
        "global_fixed_mean_utility": mean(row["global_fixed_observed_utility"] for row in rows),
        "oracle_mean_utility": mean(row["oracle_observed_utility"] for row in rows),
        "compiler_oracle_regret": mean(row["compiler_oracle_regret"] for row in rows),
        "global_fixed_oracle_regret": mean(row["global_fixed_oracle_regret"] for row in rows),
        "compiler_oracle_optimal_fraction": mean(row["compiler_oracle_optimal"] for row in rows),
        "global_fixed_oracle_optimal_fraction": mean(row["global_fixed_oracle_optimal"] for row in rows),
        "oracle_tie_rate": mean(int(int(row["oracle_set_size"]) > 1) for row in rows),
        "mean_oracle_set_size": mean(row["oracle_set_size"] for row in rows),
    }


def utility_correlations(prediction_rows: Sequence[Dict[str, Any]], methods: Sequence[str]) -> Dict[str, float | None]:
    observed = [float(row["observed_utility"]) for row in prediction_rows]
    return {
        method: pearson(
            [float(row[f"{method}_predicted_utility"]) for row in prediction_rows],
            observed,
        )
        for method in methods
    }


def lofo_summary_row(
    held_out_learning_type: str,
    model_info: Dict[str, Any],
    prediction_metrics: Dict[str, Any],
    ranking_metrics: Dict[str, Any],
    selection_summary: Dict[str, Any],
) -> Dict[str, Any]:
    primary_prediction = prediction_metrics["primary"]
    primary_ranking = ranking_metrics["primary"]
    return {
        "held_out_learning_type": held_out_learning_type,
        "n_test_episodes": selection_summary["n_test_episodes"],
        "selected_model_type": model_info["model_type"],
        "validation_mean_mae": model_info["validation_mean_mae"],
        "test_mean_mae": primary_prediction["mean_mae"],
        "utility_correlation": prediction_metrics["utility_correlation"]["primary"],
        "mean_episode_spearman": primary_ranking["mean_episode_spearman"],
        "pairwise_ranking_accuracy": primary_ranking["pairwise_ranking_accuracy"],
        "top1_oracle_recovery": primary_ranking["top1_set_oracle_recovery"],
        "top2_oracle_recovery": primary_ranking["top2_set_oracle_recovery"],
        "compiler_mean_utility": selection_summary["compiler_mean_utility"],
        "global_fixed_mean_utility": selection_summary["global_fixed_mean_utility"],
        "oracle_mean_utility": selection_summary["oracle_mean_utility"],
        "compiler_oracle_regret": selection_summary["compiler_oracle_regret"],
        "global_fixed_oracle_regret": selection_summary["global_fixed_oracle_regret"],
        "compiler_oracle_optimal_fraction": selection_summary["compiler_oracle_optimal_fraction"],
        "global_fixed_oracle_optimal_fraction": selection_summary["global_fixed_oracle_optimal_fraction"],
        "global_fixed_config_id": selection_summary["global_fixed_config_id"],
        "oracle_tie_rate": selection_summary["oracle_tie_rate"],
    }


def macro_average_row(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        raise ValueError("Cannot compute LOFO macro-average with no folds.")
    numeric_fields = [
        "validation_mean_mae",
        "test_mean_mae",
        "utility_correlation",
        "mean_episode_spearman",
        "pairwise_ranking_accuracy",
        "top1_oracle_recovery",
        "top2_oracle_recovery",
        "compiler_mean_utility",
        "global_fixed_mean_utility",
        "oracle_mean_utility",
        "compiler_oracle_regret",
        "global_fixed_oracle_regret",
        "compiler_oracle_optimal_fraction",
        "global_fixed_oracle_optimal_fraction",
        "oracle_tie_rate",
    ]
    out = {
        "held_out_learning_type": "macro_average",
        "n_test_episodes": sum(int(row["n_test_episodes"]) for row in rows),
        "selected_model_type": "",
        "global_fixed_config_id": "",
    }
    for field in numeric_fields:
        out[field] = mean(row.get(field) for row in rows)
    return out


def run_lofo_fold(
    args: argparse.Namespace,
    held_out_learning_type: str,
    dataset: Sequence[Dict[str, Any]],
    feature_names: Sequence[str],
    features: Dict[str, Dict[str, Any]],
    strict_standard_counts: bool,
) -> Dict[str, Any]:
    output_dir = Path(args.output_dir) / held_out_learning_type
    output_dir.mkdir(parents=True, exist_ok=True)
    config_ids = list(args.config_ids)

    train_rows, val_rows, test_rows = lofo_split_rows(dataset, held_out_learning_type)
    counts = validate_lofo_partition(
        train_rows,
        val_rows,
        test_rows,
        held_out_learning_type=held_out_learning_type,
        config_ids=config_ids,
        strict_standard_counts=strict_standard_counts,
    )

    standardizer = fit_standardizer([row["x"] for row in train_rows])
    for row in train_rows + val_rows + test_rows:
        row["x_std"] = apply_standardizer([row["x"]], standardizer)[0].tolist()

    model_info = train_primary_predictor(args, train_rows=train_rows, val_rows=val_rows)
    y_test = np.asarray([row["y"] for row in test_rows], dtype=float)
    predictions_by_method = {
        "primary": predict_primary(model_info, test_rows),
        "configuration_mean": config_mean_predictions(train_rows, test_rows),
    }
    for method, pred in predictions_by_method.items():
        if not np.all(np.isfinite(pred)):
            raise ValueError(f"{method} LOFO predictions contain NaN or Inf values.")

    prediction_rows = build_prediction_rows(test_rows, predictions_by_method)
    metric_rows = prediction_metric_rows(predictions_by_method, y_test)
    prediction_metrics = prediction_metric_summary(metric_rows)
    prediction_metrics["utility_correlation"] = utility_correlations(
        prediction_rows,
        methods=list(predictions_by_method),
    )
    ranking_rows = build_ranking_rows(
        prediction_rows,
        methods=list(predictions_by_method),
        config_ids=config_ids,
        tie_tolerance=args.tie_tolerance,
    )
    ranking_metrics = summarize_ranking(ranking_rows)
    global_fixed = learn_global_fixed_program(train_rows, config_ids, tie_tolerance=args.tie_tolerance)
    compiler_selection = build_compiler_selection_rows(
        prediction_rows,
        global_fixed=global_fixed,
        config_ids=config_ids,
        tie_tolerance=args.tie_tolerance,
    )
    selection_summary = summarize_compiler_selection(compiler_selection, global_fixed)

    write_csv(
        output_dir / "predictions_test.csv",
        prediction_rows,
        list(prediction_rows[0].keys()) if prediction_rows else [],
    )
    write_csv(
        output_dir / "prediction_metrics_by_outcome.csv",
        metric_rows,
        ["method", "outcome", "mae", "rmse"],
    )
    write_csv(
        output_dir / "ranking_metrics_by_episode.csv",
        ranking_rows,
        [
            "episode_id",
            "learning_type",
            "method",
            "spearman",
            "pairwise_ranking_accuracy",
            "top1_set_oracle_recovery",
            "top2_set_oracle_recovery",
            "top1_set_fractional_oracle_credit",
            "top2_set_fractional_oracle_credit",
            "selected_config",
            "selected_config_tie_break",
            "selected_config_oracle_optimal",
            "observed_oracle_set",
            "predicted_top1_set",
            "predicted_top2_set",
            "observed_oracle_utility",
            "predicted_best_utility",
        ],
    )
    write_csv(
        output_dir / "compiler_selection.csv",
        compiler_selection,
        [
            "episode_id",
            "learning_type",
            "compiler_selected_config_id",
            "compiler_predicted_top1_set",
            "compiler_predicted_top2_set",
            "global_fixed_config_id",
            "oracle_config_ids",
            "oracle_set_size",
            "compiler_observed_utility",
            "global_fixed_observed_utility",
            "oracle_observed_utility",
            "compiler_oracle_regret",
            "global_fixed_oracle_regret",
            "compiler_oracle_optimal",
            "global_fixed_oracle_optimal",
        ],
    )
    write_json(output_dir / "prediction_metrics.json", prediction_metrics)
    write_json(output_dir / "ranking_metrics.json", ranking_metrics)
    write_json(output_dir / "compiler_selection_summary.json", selection_summary)

    standardizer_payload = {
        "feature_names": list(feature_names),
        **standardizer,
        "fit_scope": "lofo_training_rows_only",
        "fit_row_count": len(train_rows),
        "fit_episode_count": counts["train_episodes"],
        "held_out_learning_type": held_out_learning_type,
        "training_learning_types": counts["training_learning_types"],
    }
    write_json(output_dir / "feature_standardizer.json", standardizer_payload)
    save_trained_model(output_dir, model_info, feature_names)

    metadata = {
        "experiment": "experiment2_lofo_generalization",
        "held_out_learning_type": held_out_learning_type,
        "training_learning_types": counts["training_learning_types"],
        "validation_learning_types": counts["validation_learning_types"],
        "test_learning_types": counts["test_learning_types"],
        "feature_set": args.feature_set,
        "model_type": model_info["model_type"],
        "candidate_model_types": model_info["candidate_model_types"],
        "selected_hyperparameters": model_info["selected_hyperparameters"],
        "validation_mean_mae": model_info["validation_mean_mae"],
        "validation_model_selection_results": model_info["validation_model_selection_results"],
        "train_rows": counts["train_rows"],
        "validation_rows": counts["validation_rows"],
        "test_rows": counts["test_rows"],
        "train_episodes": counts["train_episodes"],
        "validation_episodes": counts["validation_episodes"],
        "test_episodes": counts["test_episodes"],
        "config_ids": config_ids,
        "random_seed": args.seed,
        "objective_identity_used_in_primary_predictor": False,
        "held_out_objective_seen_during_training": False,
        "held_out_objective_seen_during_validation": False,
        "test_rows_used_for_model_fitting_or_selection": False,
        "standardizer_fit_scope": "lofo_training_rows_only",
        "geometry_source_path": str(args.geometry),
        "feature_source_path": str(args.features),
        "config_library": str(args.config_library),
        "feature_names": list(feature_names),
        "feature_standardizer_path": str(output_dir / "feature_standardizer.json"),
        "configuration_mean_baseline_fit_scope": "lofo_training_rows_only",
        "global_fixed_baseline_fit_scope": "lofo_training_rows_only",
        "global_fixed_baseline": global_fixed,
        "objective_conditioned_baseline_omitted": True,
        "representation_metadata": first_representation_metadata(features),
        "git_commit": git_commit(),
    }
    write_json(output_dir / "model_metadata.json", metadata)

    summary_row = lofo_summary_row(
        held_out_learning_type,
        model_info=model_info,
        prediction_metrics=prediction_metrics,
        ranking_metrics=ranking_metrics,
        selection_summary=selection_summary,
    )
    return {
        "summary_row": summary_row,
        "metadata": metadata,
        "prediction_metrics": prediction_metrics,
        "ranking_metrics": ranking_metrics,
        "compiler_selection_summary": selection_summary,
    }


def run_lofo(args: argparse.Namespace) -> Dict[str, Any]:
    config_ids = list(args.config_ids)
    geometry_rows = read_csv_rows(args.geometry)
    assert_no_split_leakage(geometry_rows)
    assert_seed_aggregated(geometry_rows)
    for split in ["train", "validation", "test"]:
        assert_complete_config_coverage(geometry_rows, config_ids, split=split)
    strict_standard_counts = standard_dataset_expected(geometry_rows, config_ids)

    features = read_feature_payloads(args.features)
    configs = load_config_library(args.config_library, config_ids=config_ids)
    dataset, feature_names = build_supervised_rows(
        geometry_rows,
        features=features,
        configs=configs,
        config_ids=config_ids,
        feature_set=args.feature_set,
    )
    assert_objective_identity_absent(feature_names)

    objectives = objective_list_from_geometry(geometry_rows)
    if args.held_out_learning_type:
        if args.held_out_learning_type not in objectives:
            raise ValueError(
                f"held_out_learning_type={args.held_out_learning_type!r} not found in geometry objectives {objectives}"
            )
        objectives = [args.held_out_learning_type]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fold_results = []
    summary_rows = []
    for held_out in objectives:
        result = run_lofo_fold(
            args,
            held_out_learning_type=held_out,
            dataset=[dict(row) for row in dataset],
            feature_names=feature_names,
            features=features,
            strict_standard_counts=strict_standard_counts,
        )
        fold_results.append(result)
        summary_rows.append(result["summary_row"])

    summary_with_macro = [*summary_rows, macro_average_row(summary_rows)]
    summary_fields = [
        "held_out_learning_type",
        "n_test_episodes",
        "selected_model_type",
        "validation_mean_mae",
        "test_mean_mae",
        "utility_correlation",
        "mean_episode_spearman",
        "pairwise_ranking_accuracy",
        "top1_oracle_recovery",
        "top2_oracle_recovery",
        "compiler_mean_utility",
        "global_fixed_mean_utility",
        "oracle_mean_utility",
        "compiler_oracle_regret",
        "global_fixed_oracle_regret",
        "compiler_oracle_optimal_fraction",
        "global_fixed_oracle_optimal_fraction",
        "global_fixed_config_id",
        "oracle_tie_rate",
    ]
    write_csv(output_dir / "lofo_summary.csv", summary_with_macro, summary_fields)
    write_json(
        output_dir / "lofo_summary.json",
        {
            "experiment": "experiment2_lofo_generalization",
            "feature_set": args.feature_set,
            "model_type": args.model_type,
            "config_ids": config_ids,
            "held_out_learning_types": objectives,
            "strict_standard_counts": strict_standard_counts,
            "summary": summary_with_macro,
        },
    )
    return {
        "summary": summary_with_macro,
        "fold_results": fold_results,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run leave-one-objective-out Experiment 2 generalization.")
    parser.add_argument("--geometry", required=True, help="Seed-aggregated Experiment 2 geometry dataset CSV.")
    parser.add_argument("--features", required=True, help="Episode feature JSONL/CSV/parquet file or per-episode JSON directory.")
    parser.add_argument("--config_library", default="data/compiler/config_library.jsonl")
    parser.add_argument("--feature_set", choices=["episode", "frozen_behavior", "module_probes", "full"], default="episode")
    parser.add_argument("--model_type", choices=["ridge", "random_forest", "auto"], default="auto")
    parser.add_argument("--output_dir", default="outputs/compiler/experiment2/lofo_episode")
    parser.add_argument("--config_ids", nargs="+", default=PRIMARY_CONFIGS)
    parser.add_argument("--held_out_learning_type", default=None)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--ridge_alphas", nargs="+", type=float, default=[0.0, 0.01, 0.1, 1.0, 10.0, 100.0])
    parser.add_argument("--rf_n_estimators", nargs="+", type=int, default=[100, 300])
    parser.add_argument("--rf_max_depth", nargs="+", default=["none", "6"])
    parser.add_argument("--rf_min_samples_leaf", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument("--tie_tolerance", type=float, default=1e-12)
    return parser.parse_args()


def main() -> None:
    result = run_lofo(parse_args())
    print(json.dumps({"summary": result["summary"]}, indent=2))


if __name__ == "__main__":
    main()
