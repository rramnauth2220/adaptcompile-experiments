#!/usr/bin/env python3
"""Experiment 2: pre-adaptation prediction of adaptation geometry."""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

try:  # pragma: no cover
    from .compiler_common import read_jsonl, write_csv
    from .compiler_feature_encoding import encode_episode_config
    from .prepare_geometry_dataset import OUTCOME_COLUMNS, PRIMARY_CONFIGS
except ImportError:  # pragma: no cover
    from compiler_common import read_jsonl, write_csv
    from compiler_feature_encoding import encode_episode_config
    from prepare_geometry_dataset import OUTCOME_COLUMNS, PRIMARY_CONFIGS


FORBIDDEN_FEATURE_PREFIXES = (
    "observed_",
    "acquisition",
    "transfer",
    "boundedness",
    "preservation",
    "utility",
    "oracle",
    "post_",
)


def read_csv_rows(path: str | Path) -> List[Dict[str, Any]]:
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def read_feature_payloads(path: str | Path) -> Dict[str, Dict[str, Any]]:
    path = Path(path)
    payloads: Dict[str, Dict[str, Any]] = {}

    if path.is_dir():
        for feature_path in sorted(path.rglob("*.json")):
            payload = json.loads(feature_path.read_text(encoding="utf-8"))
            episode_id = payload.get("metadata", {}).get("episode_id")
            if episode_id:
                payloads[str(episode_id)] = payload
        return payloads

    if path.suffix == ".jsonl":
        rows = read_jsonl(path)
    elif path.suffix == ".csv":
        rows = read_csv_rows(path)
    elif path.suffix == ".parquet":
        try:
            import pandas as pd
        except ModuleNotFoundError as exc:  # pragma: no cover
            raise RuntimeError("Reading parquet feature files requires pandas plus a parquet engine.") from exc
        rows = pd.read_parquet(path).to_dict(orient="records")
    else:
        raise ValueError(f"Unsupported feature file type: {path}")

    for row in rows:
        if "payload_json" in row:
            payload = json.loads(row["payload_json"])
        else:
            payload = row
        episode_id = payload.get("metadata", {}).get("episode_id") or payload.get("episode_id")
        if not episode_id:
            raise ValueError(f"Feature payload lacks episode_id: {payload}")
        payloads[str(episode_id)] = payload
    return payloads


def load_config_library(path: str | Path, config_ids: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    configs = {str(row["config_id"]): row for row in read_jsonl(path)}
    missing = [cfg for cfg in config_ids if cfg not in configs]
    if missing:
        raise ValueError(f"Config library is missing requested config_ids: {missing}")
    return {cfg: configs[cfg] for cfg in config_ids}


def as_float(row: Dict[str, Any], column: str) -> float:
    value = row.get(column)
    if value in (None, ""):
        raise ValueError(f"Missing required value for {column} in {row}")
    return float(value)


def split_episode_ids(rows: Sequence[Dict[str, Any]]) -> Dict[str, set[str]]:
    out: Dict[str, set[str]] = defaultdict(set)
    for row in rows:
        model_key = str(row.get("model_slug") or "model")
        out[f"{model_key}::{row['meta_split']}"].add(str(row["episode_id"]))
    return out


def assert_no_split_leakage(rows: Sequence[Dict[str, Any]]) -> None:
    by_model_split: Dict[str, Dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for row in rows:
        by_model_split[str(row.get("model_slug") or "model")][str(row["meta_split"])].add(str(row["episode_id"]))
    for model_slug, ids in sorted(by_model_split.items()):
        splits = sorted(ids)
        for i, split_i in enumerate(splits):
            for split_j in splits[i + 1 :]:
                overlap = ids[split_i] & ids[split_j]
                if overlap:
                    sample = sorted(overlap)[:5]
                    raise ValueError(
                        f"Episode IDs overlap for model={model_slug} between "
                        f"{split_i} and {split_j}: {sample}"
                    )


def assert_no_test_rows_used_for_training(rows: Sequence[Dict[str, Any]]) -> None:
    for row in rows:
        if str(row["record"].get("meta_split")) == "test" and row.get("used_for_model_selection"):
            raise ValueError("Test rows must not be used for training or hyperparameter selection.")


def assert_complete_config_coverage(
    rows: Sequence[Dict[str, Any]],
    config_ids: Sequence[str],
    split: str,
) -> None:
    expected = set(config_ids)
    by_episode: Dict[Tuple[str, str], set[str]] = defaultdict(set)
    for row in rows:
        if str(row["meta_split"]) == split:
            by_episode[(str(row.get("model_slug") or "model"), str(row["episode_id"]))].add(str(row["config_id"]))
    incomplete = {
        ep: sorted(expected - present)
        for ep, present in by_episode.items()
        if present != expected
    }
    if incomplete:
        first = next(iter(incomplete.items()))
        raise ValueError(f"{split} episode lacks complete config coverage: {first}")


def assert_seed_aggregated(rows: Sequence[Dict[str, Any]]) -> None:
    seen: set[Tuple[str, str, str]] = set()
    for row in rows:
        key = (str(row.get("model_slug") or "model"), str(row["episode_id"]), str(row["config_id"]))
        if key in seen:
            raise ValueError(f"Geometry must have one seed-aggregated row per episode/config; duplicate {key}")
        seen.add(key)
        if row.get("number_of_seeds") in (None, ""):
            raise ValueError(f"Geometry row lacks number_of_seeds: {key}")


def assert_feature_names_safe(feature_names: Sequence[str]) -> None:
    for name in feature_names:
        raw = name.split("__", 1)[-1]
        if "learning_type" in name:
            raise ValueError(f"learning_type leaked into primary feature matrix: {name}")
        if raw.startswith(FORBIDDEN_FEATURE_PREFIXES) or name.startswith(FORBIDDEN_FEATURE_PREFIXES):
            raise ValueError(f"Post-adaptation or target-like feature leaked into matrix: {name}")


def build_supervised_rows(
    geometry_rows: Sequence[Dict[str, Any]],
    features: Dict[str, Dict[str, Any]],
    configs: Dict[str, Dict[str, Any]],
    config_ids: Sequence[str],
    feature_set: str,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    rows: List[Dict[str, Any]] = []
    feature_names: List[str] | None = None
    config_set = set(config_ids)
    for record in geometry_rows:
        config_id = str(record["config_id"])
        if config_id not in config_set:
            continue
        episode_id = str(record["episode_id"])
        if episode_id not in features:
            raise FileNotFoundError(f"No feature payload found for episode_id={episode_id}")

        payload = features[episode_id]
        if "learning_type" in payload.get("episode_features", {}):
            raise ValueError("learning_type must be metadata only, not an episode feature.")
        x, names = encode_episode_config(payload, configs[config_id], feature_set=feature_set)
        if feature_names is None:
            feature_names = names
            assert_feature_names_safe(feature_names)
        elif feature_names != names:
            raise ValueError("Feature schema mismatch across episode/config rows.")

        y = [as_float(record, col) for col in OUTCOME_COLUMNS]
        if any(not math.isfinite(v) for v in x + y):
            raise ValueError(f"Non-finite predictor input or target for {episode_id}/{config_id}")
        rows.append({"record": record, "x": x, "y": y})

    if not rows:
        raise ValueError("No supervised rows available after joining geometry/features/configs.")
    return rows, feature_names or []


def fit_standardizer(xs: Sequence[Sequence[float]]) -> Dict[str, List[float]]:
    arr = np.asarray(xs, dtype=float)
    if arr.ndim != 2 or arr.shape[0] == 0:
        raise ValueError("Cannot fit standardizer on an empty matrix.")
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    std[std == 0] = 1.0
    return {"mean": mean.tolist(), "std": std.tolist()}


def apply_standardizer(xs: Sequence[Sequence[float]], standardizer: Dict[str, List[float]]) -> np.ndarray:
    arr = np.asarray(xs, dtype=float)
    return (arr - np.asarray(standardizer["mean"], dtype=float)) / np.asarray(standardizer["std"], dtype=float)


def split_rows(rows: Sequence[Dict[str, Any]], split: str) -> List[Dict[str, Any]]:
    return [row for row in rows if str(row["record"].get("meta_split")) == split]


class RidgeMultiOutput:
    def __init__(self, alpha: float) -> None:
        self.alpha = float(alpha)
        self.coef_: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "RidgeMultiOutput":
        x_bias = np.column_stack([np.ones(x.shape[0]), x])
        penalty = np.eye(x_bias.shape[1]) * self.alpha
        penalty[0, 0] = 0.0
        self.coef_ = np.linalg.pinv(x_bias.T @ x_bias + penalty) @ x_bias.T @ y
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("Ridge model has not been fit.")
        x_bias = np.column_stack([np.ones(x.shape[0]), x])
        return x_bias @ self.coef_


def metric_mae(pred: np.ndarray, gold: np.ndarray) -> np.ndarray:
    return np.mean(np.abs(pred - gold), axis=0)


def metric_rmse(pred: np.ndarray, gold: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean((pred - gold) ** 2, axis=0))


def choose_ridge_alpha(
    train_rows: Sequence[Dict[str, Any]],
    val_rows: Sequence[Dict[str, Any]],
    alphas: Sequence[float],
) -> Tuple[RidgeMultiOutput, float, List[Dict[str, float]]]:
    if not val_rows:
        raise ValueError("Validation rows are required for ridge hyperparameter selection.")
    x_train = np.asarray([row["x_std"] for row in train_rows], dtype=float)
    y_train = np.asarray([row["y"] for row in train_rows], dtype=float)
    x_eval = np.asarray([row["x_std"] for row in val_rows], dtype=float)
    y_eval = np.asarray([row["y"] for row in val_rows], dtype=float)

    results = []
    best: Tuple[float, float, RidgeMultiOutput] | None = None
    for alpha in alphas:
        model = RidgeMultiOutput(alpha=alpha).fit(x_train, y_train)
        pred = model.predict(x_eval)
        mean_mae = float(metric_mae(pred, y_eval).mean())
        results.append({"alpha": float(alpha), "validation_mean_mae": mean_mae})
        key = (mean_mae, float(alpha))
        if best is None or key < (best[0], best[1]):
            best = (mean_mae, float(alpha), model)
    assert best is not None
    return best[2], best[1], results


def sklearn_random_forest_available() -> bool:
    try:
        import sklearn  # noqa: F401
        from sklearn.ensemble import RandomForestRegressor  # noqa: F401
    except ModuleNotFoundError:
        return False
    return True


def choose_random_forest(
    train_rows: Sequence[Dict[str, Any]],
    val_rows: Sequence[Dict[str, Any]],
    seed: int,
    n_estimators_grid: Sequence[int],
    max_depth_grid: Sequence[str],
    min_samples_leaf_grid: Sequence[int],
) -> Tuple[Any, Dict[str, Any], List[Dict[str, Any]]]:
    if not val_rows:
        raise ValueError("Validation rows are required for random forest model selection.")
    try:
        from sklearn.ensemble import RandomForestRegressor
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "random_forest requires scikit-learn. Install sklearn or use --model_type ridge."
        ) from exc

    x_train = np.asarray([row["x"] for row in train_rows], dtype=float)
    y_train = np.asarray([row["y"] for row in train_rows], dtype=float)
    x_eval = np.asarray([row["x"] for row in val_rows], dtype=float)
    y_eval = np.asarray([row["y"] for row in val_rows], dtype=float)

    results: List[Dict[str, Any]] = []
    best: Tuple[float, str, Any, Dict[str, Any]] | None = None
    for n_estimators in n_estimators_grid:
        for max_depth_value in max_depth_grid:
            max_depth = None if str(max_depth_value).lower() == "none" else int(max_depth_value)
            for min_samples_leaf in min_samples_leaf_grid:
                params = {
                    "n_estimators": int(n_estimators),
                    "max_depth": max_depth,
                    "min_samples_leaf": int(min_samples_leaf),
                    "random_state": int(seed),
                    "n_jobs": 1,
                }
                model = RandomForestRegressor(**params)
                model.fit(x_train, y_train)
                pred = model.predict(x_eval)
                mean_mae = float(metric_mae(np.asarray(pred, dtype=float), y_eval).mean())
                result = {
                    "model_type": "random_forest",
                    **params,
                    "validation_mean_mae": mean_mae,
                }
                results.append(result)
                key = (mean_mae, json.dumps(params, sort_keys=True))
                if best is None or key < (best[0], best[1]):
                    best = (mean_mae, key[1], model, params)
    assert best is not None
    return best[2], best[3], results


def train_primary_predictor(
    args: argparse.Namespace,
    train_rows: Sequence[Dict[str, Any]],
    val_rows: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    candidate_results: List[Dict[str, Any]] = []
    candidates: List[Dict[str, Any]] = []

    if args.model_type in {"ridge", "auto"}:
        ridge_model, selected_alpha, ridge_search = choose_ridge_alpha(
            train_rows,
            val_rows,
            [float(x) for x in args.ridge_alphas],
        )
        best_ridge_score = min(row["validation_mean_mae"] for row in ridge_search)
        candidate_results.extend({"model_type": "ridge", **row} for row in ridge_search)
        candidates.append(
            {
                "model_type": "ridge",
                "model": ridge_model,
                "validation_mean_mae": best_ridge_score,
                "selected_hyperparameters": {"ridge_alpha": selected_alpha},
                "uses_standardizer": True,
            }
        )

    if args.model_type in {"random_forest", "auto"}:
        if sklearn_random_forest_available():
            rf_model, rf_params, rf_search = choose_random_forest(
                train_rows,
                val_rows,
                seed=args.seed,
                n_estimators_grid=args.rf_n_estimators,
                max_depth_grid=args.rf_max_depth,
                min_samples_leaf_grid=args.rf_min_samples_leaf,
            )
            best_rf_score = min(row["validation_mean_mae"] for row in rf_search)
            candidate_results.extend(rf_search)
            candidates.append(
                {
                    "model_type": "random_forest",
                    "model": rf_model,
                    "validation_mean_mae": best_rf_score,
                    "selected_hyperparameters": rf_params,
                    "uses_standardizer": False,
                }
            )
        elif args.model_type == "random_forest":
            raise RuntimeError(
                "random_forest was requested but scikit-learn is not installed."
            )
        else:
            candidate_results.append(
                {
                    "model_type": "random_forest",
                    "supported": False,
                    "reason": "scikit-learn is not installed",
                }
            )

    if not candidates:
        raise ValueError("No supported primary predictor candidates were available.")

    selected = sorted(
        candidates,
        key=lambda row: (float(row["validation_mean_mae"]), str(row["model_type"])),
    )[0]
    selected["candidate_model_types"] = [row["model_type"] for row in candidates]
    selected["validation_model_selection_results"] = candidate_results
    return selected


def predict_primary(model_info: Dict[str, Any], rows: Sequence[Dict[str, Any]]) -> np.ndarray:
    key = "x_std" if model_info.get("uses_standardizer") else "x"
    x = np.asarray([row[key] for row in rows], dtype=float)
    pred = np.asarray(model_info["model"].predict(x), dtype=float)
    if pred.ndim == 1:
        pred = pred.reshape(-1, 1)
    if not np.all(np.isfinite(pred)):
        raise ValueError("Primary predictor produced NaN or Inf values.")
    return pred


def config_mean_predictions(
    train_rows: Sequence[Dict[str, Any]],
    eval_rows: Sequence[Dict[str, Any]],
) -> np.ndarray:
    by_config: Dict[str, List[List[float]]] = defaultdict(list)
    for row in train_rows:
        by_config[str(row["record"]["config_id"])].append(row["y"])
    global_mean = np.asarray([row["y"] for row in train_rows], dtype=float).mean(axis=0)
    means = {
        config_id: np.asarray(values, dtype=float).mean(axis=0)
        for config_id, values in by_config.items()
    }
    return np.vstack([
        means.get(str(row["record"]["config_id"]), global_mean)
        for row in eval_rows
    ])


def objective_config_mean_predictions(
    train_rows: Sequence[Dict[str, Any]],
    eval_rows: Sequence[Dict[str, Any]],
) -> np.ndarray:
    by_key: Dict[Tuple[str, str], List[List[float]]] = defaultdict(list)
    for row in train_rows:
        rec = row["record"]
        by_key[(str(rec["learning_type"]), str(rec["config_id"]))].append(row["y"])
    fallback = config_mean_predictions(train_rows, eval_rows)
    means = {
        key: np.asarray(values, dtype=float).mean(axis=0)
        for key, values in by_key.items()
    }
    rows = []
    for i, row in enumerate(eval_rows):
        rec = row["record"]
        rows.append(means.get((str(rec["learning_type"]), str(rec["config_id"])), fallback[i]))
    return np.vstack(rows)


def ranks(values: Sequence[float], tolerance: float) -> List[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    out = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and abs(indexed[j + 1][1] - indexed[i][1]) <= tolerance:
            j += 1
        rank = (i + j) / 2.0
        for k in range(i, j + 1):
            out[indexed[k][0]] = rank
        i = j + 1
    return out


def pearson(a: Sequence[float], b: Sequence[float]) -> float | None:
    if len(a) < 2:
        return None
    av = np.asarray(a, dtype=float)
    bv = np.asarray(b, dtype=float)
    if float(av.std()) == 0.0 or float(bv.std()) == 0.0:
        return None
    return float(np.corrcoef(av, bv)[0, 1])


def spearman(a: Sequence[float], b: Sequence[float], tolerance: float) -> float | None:
    return pearson(ranks(a, tolerance), ranks(b, tolerance))


def utility(values: Sequence[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=float)))


def sign_with_tolerance(value: float, tolerance: float) -> int:
    if value > tolerance:
        return 1
    if value < -tolerance:
        return -1
    return 0


def top_k_set(configs: Sequence[str], values: Sequence[float], k: int, tolerance: float) -> List[str]:
    ordered = sorted(zip(configs, values), key=lambda item: (-item[1], item[0]))
    if not ordered:
        return []
    boundary_idx = min(max(k, 1), len(ordered)) - 1
    boundary = ordered[boundary_idx][1]
    return sorted(cfg for cfg, value in ordered if value >= boundary - tolerance)


def selected_config_with_lexicographic_tie_break(
    configs: Sequence[str],
    values: Sequence[float],
    tolerance: float,
) -> str:
    top = top_k_set(configs, values, 1, tolerance)
    if not top:
        raise ValueError("Cannot select a config from an empty prediction set.")
    return sorted(top)[0]


def rank_episode(
    rows: Sequence[Dict[str, Any]],
    method: str,
    tolerance: float,
) -> Dict[str, Any]:
    configs = [str(row["config_id"]) for row in rows]
    observed = [float(row["observed_utility"]) for row in rows]
    predicted = [float(row[f"{method}_predicted_utility"]) for row in rows]
    max_observed = max(observed)
    oracle_set = sorted(cfg for cfg, value in zip(configs, observed) if value >= max_observed - tolerance)
    pred_top1 = top_k_set(configs, predicted, 1, tolerance)
    pred_top2 = top_k_set(configs, predicted, 2, tolerance)
    selected_config = selected_config_with_lexicographic_tie_break(configs, predicted, tolerance)
    top1_intersection = set(pred_top1) & set(oracle_set)
    top2_intersection = set(pred_top2) & set(oracle_set)

    pair_hits = 0
    pair_total = 0
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            pred_sign = sign_with_tolerance(predicted[i] - predicted[j], tolerance)
            obs_sign = sign_with_tolerance(observed[i] - observed[j], tolerance)
            pair_total += 1
            pair_hits += int(pred_sign == obs_sign)

    return {
        "episode_id": rows[0]["episode_id"],
        "learning_type": rows[0]["learning_type"],
        "method": method,
        "spearman": spearman(predicted, observed, tolerance),
        "pairwise_ranking_accuracy": pair_hits / pair_total if pair_total else None,
        "top1_set_oracle_recovery": bool(top1_intersection),
        "top2_set_oracle_recovery": bool(top2_intersection),
        "top1_set_fractional_oracle_credit": len(top1_intersection) / len(pred_top1) if pred_top1 else None,
        "top2_set_fractional_oracle_credit": len(top2_intersection) / len(pred_top2) if pred_top2 else None,
        "selected_config": selected_config,
        "selected_config_tie_break": "lexicographic config_id among predicted top-tied configs",
        "selected_config_oracle_optimal": selected_config in set(oracle_set),
        "observed_oracle_set": "|".join(oracle_set),
        "predicted_top1_set": "|".join(pred_top1),
        "predicted_top2_set": "|".join(pred_top2),
        "observed_oracle_utility": max_observed,
        "predicted_best_utility": max(predicted),
    }


def summarize_ranking(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    by_method: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_method[str(row["method"])].append(row)
    summary = {}
    for method, method_rows in sorted(by_method.items()):
        spearmans = [float(row["spearman"]) for row in method_rows if row["spearman"] is not None]
        pairwise = [float(row["pairwise_ranking_accuracy"]) for row in method_rows if row["pairwise_ranking_accuracy"] is not None]
        summary[method] = {
            "mean_episode_spearman": float(np.mean(spearmans)) if spearmans else None,
            "pairwise_ranking_accuracy": float(np.mean(pairwise)) if pairwise else None,
            "top1_set_oracle_recovery": float(np.mean([row["top1_set_oracle_recovery"] for row in method_rows])),
            "top2_set_oracle_recovery": float(np.mean([row["top2_set_oracle_recovery"] for row in method_rows])),
            "top1_set_fractional_oracle_credit": float(
                np.mean([row["top1_set_fractional_oracle_credit"] for row in method_rows])
            ),
            "top2_set_fractional_oracle_credit": float(
                np.mean([row["top2_set_fractional_oracle_credit"] for row in method_rows])
            ),
            "selected_config_oracle_optimal": float(np.mean([row["selected_config_oracle_optimal"] for row in method_rows])),
            "n_episodes": len(method_rows),
        }
    return summary


def build_prediction_rows(
    eval_rows: Sequence[Dict[str, Any]],
    predictions_by_method: Dict[str, np.ndarray],
) -> List[Dict[str, Any]]:
    out = []
    for i, row in enumerate(eval_rows):
        rec = row["record"]
        base = {
            "model_slug": rec.get("model_slug"),
            "episode_id": rec["episode_id"],
            "learning_type": rec["learning_type"],
            "meta_split": rec["meta_split"],
            "config_id": rec["config_id"],
            "number_of_seeds": rec.get("number_of_seeds"),
            "seed_values": rec.get("seed_values"),
        }
        for j, col in enumerate(OUTCOME_COLUMNS):
            base[f"observed_{col}"] = float(row["y"][j])
        base["observed_utility"] = utility(row["y"])

        primary = predictions_by_method["primary"]
        for j, col in enumerate(OUTCOME_COLUMNS):
            base[f"predicted_{col}"] = float(primary[i, j])
        base["predicted_utility"] = utility(primary[i])

        for method, pred in predictions_by_method.items():
            for j, col in enumerate(OUTCOME_COLUMNS):
                base[f"{method}_predicted_{col}"] = float(pred[i, j])
            base[f"{method}_predicted_utility"] = utility(pred[i])
        out.append(base)
    return out


def prediction_metric_rows(
    predictions_by_method: Dict[str, np.ndarray],
    gold: np.ndarray,
) -> List[Dict[str, Any]]:
    rows = []
    for method, pred in predictions_by_method.items():
        maes = metric_mae(pred, gold)
        rmses = metric_rmse(pred, gold)
        for i, outcome in enumerate(OUTCOME_COLUMNS):
            rows.append(
                {
                    "method": method,
                    "outcome": outcome,
                    "mae": float(maes[i]),
                    "rmse": float(rmses[i]),
                }
            )
    return rows


def prediction_metric_summary(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    by_method: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_method[str(row["method"])].append(row)
    return {
        method: {
            "mean_mae": float(np.mean([float(row["mae"]) for row in method_rows])),
            "mean_rmse": float(np.mean([float(row["rmse"]) for row in method_rows])),
            "by_outcome": {
                row["outcome"]: {"mae": float(row["mae"]), "rmse": float(row["rmse"])}
                for row in method_rows
            },
        }
        for method, method_rows in sorted(by_method.items())
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


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def first_representation_metadata(features: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    keys = [
        "episode_representation_backend",
        "episode_representation_source_layer",
        "episode_representation_token_pooling",
        "episode_representation_episode_pooling",
        "episode_representation_projection_dim",
        "episode_representation_projection_seed",
    ]
    for payload in features.values():
        metadata = payload.get("metadata", {})
        if any(key in metadata for key in keys):
            return {key: metadata.get(key) for key in keys if key in metadata}
    return {}


def save_trained_model(
    output_dir: Path,
    model_info: Dict[str, Any],
    feature_names: Sequence[str],
) -> None:
    model_type = str(model_info["model_type"])
    model = model_info["model"]
    if model_type == "ridge":
        coef = model.coef_
        np.savez(output_dir / "trained_model_ridge.npz", coef=coef)
        (output_dir / "ridge_coefficients.json").write_text(
            json.dumps(
                {
                    "feature_names": ["intercept", *feature_names],
                    "outcome_columns": OUTCOME_COLUMNS,
                    "coef": coef.tolist() if coef is not None else None,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return
    with (output_dir / f"trained_model_{model_type}.pkl").open("wb") as f:
        pickle.dump(model, f)


def run_experiment2(args: argparse.Namespace) -> Dict[str, Any]:
    config_ids = list(args.config_ids)
    geometry_rows = read_csv_rows(args.geometry)
    assert_no_split_leakage(geometry_rows)
    assert_seed_aggregated(geometry_rows)
    for split in ["train", "validation", "test"]:
        assert_complete_config_coverage(geometry_rows, config_ids, split=split)

    features = read_feature_payloads(args.features)
    configs = load_config_library(args.config_library, config_ids=config_ids)
    dataset, feature_names = build_supervised_rows(
        geometry_rows,
        features=features,
        configs=configs,
        config_ids=config_ids,
        feature_set=args.feature_set,
    )

    train_rows = split_rows(dataset, "train")
    val_rows = split_rows(dataset, "validation")
    test_rows = split_rows(dataset, "test")
    if not train_rows:
        raise ValueError("Experiment 2 requires meta_split=train rows.")
    if not val_rows and not args.allow_train_only_smoke_test:
        raise ValueError(
            "Experiment 2 requires meta_split=validation rows for model selection. "
            "Use --allow_train_only_smoke_test only for tiny plumbing tests."
        )
    if not val_rows and args.allow_train_only_smoke_test:
        val_rows = list(train_rows)
    if not test_rows:
        raise ValueError("Experiment 2 requires meta_split=test rows for final evaluation.")
    assert_no_test_rows_used_for_training(train_rows)
    assert_no_test_rows_used_for_training(val_rows)

    standardizer = fit_standardizer([row["x"] for row in train_rows])
    for row in dataset:
        row["x_std"] = apply_standardizer([row["x"]], standardizer)[0].tolist()

    model_info = train_primary_predictor(args, train_rows=train_rows, val_rows=val_rows)
    y_test = np.asarray([row["y"] for row in test_rows], dtype=float)

    predictions_by_method = {
        "primary": predict_primary(model_info, test_rows),
        "configuration_mean": config_mean_predictions(train_rows, test_rows),
        "objective_conditioned_mean": objective_config_mean_predictions(train_rows, test_rows),
    }
    for method, pred in predictions_by_method.items():
        if not np.all(np.isfinite(pred)):
            raise ValueError(f"{method} predictions contain NaN or Inf values.")

    prediction_rows = build_prediction_rows(test_rows, predictions_by_method)
    metric_rows = prediction_metric_rows(predictions_by_method, y_test)
    prediction_metrics = prediction_metric_summary(metric_rows)
    prediction_metrics["utility_correlation"] = utility_correlations(
        prediction_rows,
        methods=list(predictions_by_method),
    )

    ranking_rows = []
    by_episode: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in prediction_rows:
        by_episode[str(row["episode_id"])].append(row)
    for episode_rows in by_episode.values():
        if {row["config_id"] for row in episode_rows} != set(config_ids):
            raise ValueError(f"Test episode lacks all configs: {episode_rows[0]['episode_id']}")
        for method in predictions_by_method:
            ranking_rows.append(rank_episode(episode_rows, method=method, tolerance=args.tie_tolerance))
    ranking_metrics = summarize_ranking(ranking_rows)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
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
    (output_dir / "prediction_metrics.json").write_text(
        json.dumps(prediction_metrics, indent=2),
        encoding="utf-8",
    )
    (output_dir / "ranking_metrics.json").write_text(
        json.dumps(ranking_metrics, indent=2),
        encoding="utf-8",
    )
    metadata = {
        "experiment": "experiment2_geometry_prediction",
        "feature_set": args.feature_set,
        "model_type": model_info["model_type"],
        "candidate_model_types": model_info["candidate_model_types"],
        "selected_hyperparameters": model_info["selected_hyperparameters"],
        "validation_model_selection_results": model_info["validation_model_selection_results"],
        "train_rows": len(train_rows),
        "validation_rows": len(val_rows),
        "test_rows": len(test_rows),
        "train_episodes": len({row["record"]["episode_id"] for row in train_rows}),
        "validation_episodes": len({row["record"]["episode_id"] for row in val_rows}),
        "test_episodes": len({row["record"]["episode_id"] for row in test_rows}),
        "config_ids": config_ids,
        "random_seed": args.seed,
        "geometry_source_path": str(args.geometry),
        "feature_source_path": str(args.features),
        "config_library": str(args.config_library),
        "feature_names": feature_names,
        "feature_standardizer_path": str(output_dir / "feature_standardizer.json"),
        "objective_identity_used_in_primary_predictor": False,
        "objective_conditioned_mean_is_diagnostic_only": True,
        "representation_metadata": first_representation_metadata(features),
        "git_commit": git_commit(),
    }
    (output_dir / "model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (output_dir / "feature_standardizer.json").write_text(
        json.dumps({"feature_names": feature_names, **standardizer}, indent=2),
        encoding="utf-8",
    )
    save_trained_model(output_dir, model_info, feature_names)

    return {
        "prediction_metrics": prediction_metrics,
        "ranking_metrics": ranking_metrics,
        "metadata": metadata,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/evaluate Experiment 2 geometry predictor.")
    parser.add_argument("--geometry", required=True, help="Seed-aggregated geometry dataset CSV.")
    parser.add_argument("--features", required=True, help="Episode feature JSONL/CSV/parquet file or per-episode JSON directory.")
    parser.add_argument("--config_library", default="data/compiler/config_library.jsonl")
    parser.add_argument("--feature_set", choices=["episode", "frozen_behavior", "module_probes", "full"], default="full")
    parser.add_argument("--model_type", choices=["ridge", "random_forest", "auto"], default="auto")
    parser.add_argument("--output_dir", default="outputs/compiler/experiment2/full")
    parser.add_argument("--config_ids", nargs="+", default=PRIMARY_CONFIGS)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--ridge_alphas", nargs="+", type=float, default=[0.0, 0.01, 0.1, 1.0, 10.0, 100.0])
    parser.add_argument("--rf_n_estimators", nargs="+", type=int, default=[100, 300])
    parser.add_argument("--rf_max_depth", nargs="+", default=["none", "6"])
    parser.add_argument("--rf_min_samples_leaf", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument("--allow_train_only_smoke_test", action="store_true")
    parser.add_argument("--tie_tolerance", type=float, default=1e-12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_experiment2(args)
    print(json.dumps({"prediction_metrics": result["prediction_metrics"], "ranking_metrics": result["ranking_metrics"]}, indent=2))


if __name__ == "__main__":
    main()
