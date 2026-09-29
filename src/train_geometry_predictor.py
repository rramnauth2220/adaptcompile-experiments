#!/usr/bin/env python3
"""Train a small MLP to predict adaptation geometry outcomes."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

try:  # Torch is required only for actual MLP training, not helper tests.
    import torch
    from torch import nn
except ModuleNotFoundError:  # pragma: no cover - depends on local environment
    torch = None
    nn = None

try:  # pragma: no cover
    from .compiler_common import read_jsonl, write_csv
    from .compiler_feature_encoding import encode_episode_config
except ImportError:  # pragma: no cover
    from compiler_common import read_jsonl, write_csv
    from compiler_feature_encoding import encode_episode_config


TARGET_COLUMNS = ["acquisition", "transfer", "boundedness"]


def read_table(path: str | Path) -> List[Dict[str, Any]]:
    path = Path(path)
    if path.suffix == ".jsonl":
        return read_jsonl(path)
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def safe_episode_path(episode_id: str) -> str:
    return episode_id.replace("::", "__").replace("/", "_")


def load_feature_payload(features_root: Path, model_slug: str, episode_id: str) -> Dict[str, Any]:
    candidates = [
        features_root / model_slug / f"{safe_episode_path(episode_id)}.json",
        features_root / f"{safe_episode_path(episode_id)}.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return json.loads(candidate.read_text(encoding="utf-8"))
    raise FileNotFoundError(f"No feature file found for {model_slug}/{episode_id} under {features_root}")


def load_config_library(path: str | Path) -> Dict[str, Dict[str, Any]]:
    return {row["config_id"]: row for row in read_jsonl(path)}


def target_values(row: Dict[str, Any], target_columns: Sequence[str]) -> List[float] | None:
    values: List[float] = []
    for col in target_columns:
        value = row.get(col)
        if value in (None, ""):
            return None
        values.append(float(value))
    return values


def build_dataset(
    records: Sequence[Dict[str, Any]],
    features_root: Path,
    configs: Dict[str, Dict[str, Any]],
    target_columns: Sequence[str] = TARGET_COLUMNS,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    rows: List[Dict[str, Any]] = []
    feature_names: List[str] | None = None
    for record in records:
        y = target_values(record, target_columns)
        if y is None:
            continue
        config_id = record["config_id"]
        if config_id not in configs:
            continue
        payload = load_feature_payload(
            features_root,
            str(record.get("model_slug") or "model"),
            str(record["episode_id"]),
        )
        x, names = encode_episode_config(payload, configs[config_id])
        if feature_names is None:
            feature_names = names
        elif feature_names != names:
            raise ValueError("Feature name mismatch across encoded records")
        rows.append(
            {
                "record": record,
                "x": x,
                "y": y,
            }
        )
    return rows, feature_names or []


def fit_standardizer(xs: Sequence[Sequence[float]]) -> Dict[str, List[float]]:
    if not xs:
        raise ValueError("Cannot fit standardizer with no rows")
    n_features = len(xs[0])
    means = []
    stds = []
    for j in range(n_features):
        vals = [float(row[j]) for row in xs]
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / len(vals)
        std = math.sqrt(var) or 1.0
        means.append(mean)
        stds.append(std)
    return {"mean": means, "std": stds}


def apply_standardizer(xs: Sequence[Sequence[float]], standardizer: Dict[str, List[float]]) -> List[List[float]]:
    means = standardizer["mean"]
    stds = standardizer["std"]
    return [
        [(float(x[j]) - means[j]) / stds[j] for j in range(len(means))]
        for x in xs
    ]


if nn is not None:
    class GeometryMLP(nn.Module):
        def __init__(self, n_features: int, n_outputs: int, hidden_dim: int = 64):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(n_features, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, n_outputs),
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return self.net(x)
else:  # pragma: no cover
    class GeometryMLP:  # type: ignore[no-redef]
        pass


def mae(pred: Sequence[Sequence[float]], gold: Sequence[Sequence[float]], columns: Sequence[str]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if not pred:
        return {col: float("nan") for col in columns}
    for j, col in enumerate(columns):
        out[col] = sum(abs(float(p[j]) - float(g[j])) for p, g in zip(pred, gold)) / len(pred)
    return out


def ranks(values: Sequence[float]) -> List[float]:
    indexed = sorted(enumerate(values), key=lambda x: x[1])
    out = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        rank = (i + j) / 2.0
        for k in range(i, j + 1):
            out[indexed[k][0]] = rank
        i = j + 1
    return out


def pearson(a: Sequence[float], b: Sequence[float]) -> float | None:
    if len(a) < 2 or len(b) < 2:
        return None
    ma = sum(a) / len(a)
    mb = sum(b) / len(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    da = math.sqrt(sum((x - ma) ** 2 for x in a))
    db = math.sqrt(sum((y - mb) ** 2 for y in b))
    if da == 0 or db == 0:
        return None
    return num / (da * db)


def spearman(a: Sequence[float], b: Sequence[float]) -> float | None:
    return pearson(ranks(a), ranks(b))


def utility(values: Sequence[float]) -> float:
    return sum(float(v) for v in values) / len(values)


def ranking_metrics(pred_rows: Sequence[Dict[str, Any]], target_columns: Sequence[str]) -> Dict[str, float]:
    by_episode: Dict[Tuple[str, int], List[Dict[str, Any]]] = defaultdict(list)
    for row in pred_rows:
        by_episode[(row["episode_id"], int(row["seed"]))].append(row)

    spearmans: List[float] = []
    pairwise_hits = 0
    pairwise_total = 0
    for rows in by_episode.values():
        if len(rows) < 2:
            continue
        pred_u = [utility([float(r[f"predicted_{c}"]) for c in target_columns]) for r in rows]
        obs_u = [utility([float(r[c]) for c in target_columns]) for r in rows]
        sp = spearman(pred_u, obs_u)
        if sp is not None:
            spearmans.append(sp)
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                pred_cmp = pred_u[i] - pred_u[j]
                obs_cmp = obs_u[i] - obs_u[j]
                if pred_cmp == 0 or obs_cmp == 0:
                    continue
                pairwise_total += 1
                if (pred_cmp > 0) == (obs_cmp > 0):
                    pairwise_hits += 1
    return {
        "mean_episode_spearman": sum(spearmans) / len(spearmans) if spearmans else float("nan"),
        "pairwise_ranking_accuracy": pairwise_hits / pairwise_total if pairwise_total else float("nan"),
        "n_ranked_episode_seed_groups": len(spearmans),
    }


def split_rows(dataset: Sequence[Dict[str, Any]], split: str) -> List[Dict[str, Any]]:
    return [row for row in dataset if row["record"].get("meta_split") == split]


def train_model(
    train_rows: Sequence[Dict[str, Any]],
    val_rows: Sequence[Dict[str, Any]],
    n_features: int,
    n_outputs: int,
    seed: int,
    epochs: int,
    hidden_dim: int,
    learning_rate: float,
) -> GeometryMLP:
    if torch is None or nn is None:
        raise RuntimeError("PyTorch is required to train the geometry predictor.")
    random.seed(seed)
    torch.manual_seed(seed)
    model = GeometryMLP(n_features=n_features, n_outputs=n_outputs, hidden_dim=hidden_dim)
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()
    x_train = torch.tensor([row["x_std"] for row in train_rows], dtype=torch.float32)
    y_train = torch.tensor([row["y"] for row in train_rows], dtype=torch.float32)
    best_state = None
    best_loss = float("inf")
    eval_rows = list(val_rows) if val_rows else list(train_rows)
    x_eval = torch.tensor([row["x_std"] for row in eval_rows], dtype=torch.float32)
    y_eval = torch.tensor([row["y"] for row in eval_rows], dtype=torch.float32)
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        loss = loss_fn(model(x_train), y_train)
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(x_eval), y_eval).item())
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def predict(model: GeometryMLP, rows: Sequence[Dict[str, Any]]) -> List[List[float]]:
    if torch is None:
        raise RuntimeError("PyTorch is required to run the geometry predictor.")
    if not rows:
        return []
    model.eval()
    x = torch.tensor([row["x_std"] for row in rows], dtype=torch.float32)
    with torch.no_grad():
        return model(x).cpu().tolist()


def global_mean_baseline(train_rows: Sequence[Dict[str, Any]], eval_rows: Sequence[Dict[str, Any]]) -> List[List[float]]:
    n_outputs = len(train_rows[0]["y"])
    means = [
        sum(row["y"][j] for row in train_rows) / len(train_rows)
        for j in range(n_outputs)
    ]
    return [list(means) for _ in eval_rows]


def config_mean_baseline(train_rows: Sequence[Dict[str, Any]], eval_rows: Sequence[Dict[str, Any]]) -> List[List[float]]:
    n_outputs = len(train_rows[0]["y"])
    by_config: Dict[str, List[List[float]]] = defaultdict(list)
    for row in train_rows:
        by_config[row["record"]["config_id"]].append(row["y"])
    config_means = {
        config_id: [
            sum(y[j] for y in ys) / len(ys)
            for j in range(n_outputs)
        ]
        for config_id, ys in by_config.items()
    }
    global_pred = global_mean_baseline(train_rows, [eval_rows[0]])[0] if eval_rows else [0.0] * n_outputs
    return [config_means.get(row["record"]["config_id"], global_pred) for row in eval_rows]


def prediction_rows(
    rows: Sequence[Dict[str, Any]],
    pred: Sequence[Sequence[float]],
    target_columns: Sequence[str],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row, yhat in zip(rows, pred):
        record = dict(row["record"])
        for col, value in zip(target_columns, yhat):
            record[f"predicted_{col}"] = float(value)
        out.append(record)
    return out


def run_training(args: argparse.Namespace) -> Dict[str, Any]:
    configs = load_config_library(args.config_library)
    records = read_table(args.records)
    target_columns = list(TARGET_COLUMNS)
    if args.include_preservation:
        target_columns.append("preservation")
    dataset, feature_names = build_dataset(
        records,
        features_root=Path(args.features_root),
        configs=configs,
        target_columns=target_columns,
    )
    if not dataset:
        raise ValueError("No usable geometry records with matching features/configs.")

    train_rows = split_rows(dataset, "train")
    val_rows = split_rows(dataset, "validation")
    test_rows = split_rows(dataset, "test")
    if not train_rows:
        raise ValueError("No meta_split=train rows available.")
    if not test_rows:
        test_rows = val_rows or train_rows

    standardizer = fit_standardizer([row["x"] for row in train_rows])
    for row in dataset:
        row["x_std"] = apply_standardizer([row["x"]], standardizer)[0]

    model = train_model(
        train_rows,
        val_rows,
        n_features=len(feature_names),
        n_outputs=len(target_columns),
        seed=args.seed,
        epochs=args.epochs,
        hidden_dim=args.hidden_dim,
        learning_rate=args.learning_rate,
    )

    test_pred = predict(model, test_rows)
    test_gold = [row["y"] for row in test_rows]
    test_prediction_rows = prediction_rows(test_rows, test_pred, target_columns)

    global_pred = global_mean_baseline(train_rows, test_rows)
    config_pred = config_mean_baseline(train_rows, test_rows)
    metrics = {
        "mlp_mae": mae(test_pred, test_gold, target_columns),
        "global_mean_mae": mae(global_pred, test_gold, target_columns),
        "configuration_only_mae": mae(config_pred, test_gold, target_columns),
        "mlp_ranking": ranking_metrics(test_prediction_rows, target_columns),
        "n_train": len(train_rows),
        "n_validation": len(val_rows),
        "n_test": len(test_rows),
        "target_columns": target_columns,
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if torch is None:
        raise RuntimeError("PyTorch is required to save the geometry predictor.")
    torch.save(model.state_dict(), output_dir / "geometry_predictor.pt")
    (output_dir / "feature_normalization.json").write_text(
        json.dumps({"feature_names": feature_names, **standardizer}, indent=2),
        encoding="utf-8",
    )
    (output_dir / "training_metadata.json").write_text(
        json.dumps(
            {
                "records": str(args.records),
                "features_root": str(args.features_root),
                "config_library": str(args.config_library),
                "seed": args.seed,
                "epochs": args.epochs,
                "hidden_dim": args.hidden_dim,
                "objective_identity_used_as_input": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    write_csv(
        output_dir / "predictions_test.csv",
        test_prediction_rows,
        list(test_prediction_rows[0].keys()) if test_prediction_rows else [],
    )
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train adaptation geometry predictor.")
    parser.add_argument("--records", default="outputs/compiler/geometry_records.csv")
    parser.add_argument("--features_root", default="outputs/compiler/features")
    parser.add_argument("--config_library", default="data/compiler/config_library.jsonl")
    parser.add_argument("--output_dir", default="outputs/compiler/predictor")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--include_preservation", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = run_training(args)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
