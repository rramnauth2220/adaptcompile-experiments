#!/usr/bin/env python3
"""Configuration-conditioned feature construction for compiler prediction."""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Sequence, Tuple

try:  # pragma: no cover
    from .compiler_common import approximate_parameter_cost, resolve_layers
except ImportError:  # pragma: no cover
    from compiler_common import approximate_parameter_cost, resolve_layers


FORBIDDEN_FEATURE_KEYS = {"learning_type", "objective", "objective_id"}
MODULE_FAMILIES = ["all", "attention", "mlp", "custom"]
FEATURE_SETS = {"episode", "frozen_behavior", "module_probes", "full"}


def mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if v is not None]
    return sum(vals) / len(vals) if vals else 0.0


def std(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if v is not None]
    if len(vals) <= 1:
        return 0.0
    m = mean(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / len(vals))


def feature_group_for_episode_key(key: str) -> str:
    if key.startswith("frozen_"):
        return "frozen_behavior"
    return "episode"


def include_group(feature_set: str, group: str) -> bool:
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"Unknown feature_set={feature_set!r}; expected one of {sorted(FEATURE_SETS)}")
    return feature_set == "full" or feature_set == group


def numeric_episode_features(
    episode_features: Dict[str, Any],
    feature_set: str = "full",
) -> List[Tuple[str, float]]:
    features = episode_features.get("episode_features", episode_features)
    rows: List[Tuple[str, float]] = []
    for key in sorted(features):
        if key in FORBIDDEN_FEATURE_KEYS:
            raise ValueError(f"Forbidden feature key present: {key}")
        if not include_group(feature_set, feature_group_for_episode_key(key)):
            continue
        value = features[key]
        if isinstance(value, bool):
            rows.append((f"episode__{key}", float(value)))
        elif isinstance(value, (int, float)) and value is not None:
            rows.append((f"episode__{key}", float(value)))
    return rows


def module_is_selected(module_row: Dict[str, Any], selected_layers: set[int] | None, target_modules: set[str]) -> bool:
    layer_ok = selected_layers is None or int(module_row["layer_index"]) in selected_layers
    module_ok = str(module_row["module_type"]) in target_modules
    return layer_ok and module_ok


def numeric_values(rows: Sequence[Dict[str, Any]], *keys: str) -> List[float]:
    values: List[float] = []
    for row in rows:
        for key in keys:
            if row.get(key) not in (None, ""):
                values.append(float(row[key]))
                break
    return values


def mean_std_features(prefix: str, name: str, values: Sequence[float]) -> List[Tuple[str, float]]:
    return [
        (f"{prefix}__{name}_mean", mean(values)),
        (f"{prefix}__{name}_std", std(values)),
    ]


def aggregate_module_rows(prefix: str, rows: Sequence[Dict[str, Any]]) -> List[Tuple[str, float]]:
    sensitivity = numeric_values(rows, "sensitivity_mean", "sensitivity_proxy")
    gradient_magnitude = numeric_values(
        rows,
        "gradient_magnitude_mean",
        "gradient_magnitude_proxy",
    )
    activation_rms = numeric_values(rows, "activation_rms_mean", "activation_rms_proxy")
    agreement = numeric_values(rows, "gradient_agreement_proxy")
    depths = numeric_values(rows, "normalized_depth")
    features: List[Tuple[str, float]] = [
        (f"{prefix}__n_modules", float(len(rows))),
    ]
    features.extend(mean_std_features(prefix, "sensitivity", sensitivity))
    features.extend(mean_std_features(prefix, "gradient_magnitude", gradient_magnitude))
    features.extend(mean_std_features(prefix, "activation_rms", activation_rms))
    features.extend(mean_std_features(prefix, "agreement", agreement))
    features.extend(mean_std_features(prefix, "depth", depths))
    return [
        *features,
        (f"{prefix}__depth_min", min(depths) if depths else 0.0),
        (f"{prefix}__depth_max", max(depths) if depths else 0.0),
    ]


def selected_layers_for_config(config: Dict[str, Any], n_layers: int) -> set[int] | None:
    condition = config.get("localization_condition", "full")
    if condition == "full":
        return None
    layers = config.get("resolved_layer_indices")
    if layers is None:
        layers = config.get("layer_indices")
    if layers is None:
        layers = resolve_layers(
            condition,
            n_layers=n_layers,
            region_width=int(config.get("region_width", max(1, n_layers // 4))),
        )
    return set(int(x) for x in layers)


def layer_summary_for_config(config: Dict[str, Any], n_layers: int) -> Dict[str, float]:
    selected_layers = selected_layers_for_config(config, n_layers=n_layers)
    if selected_layers is None:
        layers = list(range(n_layers))
        is_full = 1.0
    else:
        layers = sorted(selected_layers)
        is_full = 0.0
    depths = [layer / max(1, n_layers - 1) for layer in layers]
    return {
        "is_full_stack": is_full,
        "is_localized": 1.0 - is_full,
        "n_layers_adapted": float(len(layers)),
        "fraction_layers_adapted": float(len(layers) / n_layers) if n_layers else 0.0,
        "depth_mean": mean(depths),
        "depth_std": std(depths),
        "depth_min": min(depths) if depths else 0.0,
        "depth_max": max(depths) if depths else 0.0,
    }


def validate_module_selection(
    config: Dict[str, Any],
    module_rows: Sequence[Dict[str, Any]],
    selected: Sequence[Dict[str, Any]],
    target_modules: set[str],
) -> None:
    config_id = str(config.get("config_id", "<unknown>"))
    if not target_modules:
        raise ValueError(
            f"Config {config_id} has no target_modules; cannot build configuration-conditioned features."
        )
    if module_rows and not selected:
        available = sorted({str(row.get("module_type")) for row in module_rows})
        condition = config.get("localization_condition")
        layers = config.get("resolved_layer_indices", config.get("layer_indices"))
        raise ValueError(
            f"Config {config_id} selected zero probe modules. "
            f"condition={condition!r}, layers={layers!r}, "
            f"target_modules={sorted(target_modules)!r}, available_module_types={available!r}"
        )


def encode_episode_config(
    episode_features: Dict[str, Any],
    config: Dict[str, Any],
    feature_set: str = "full",
) -> Tuple[List[float], List[str]]:
    """Return a fixed feature vector for an episode/config pair.

    `learning_type` is allowed in metadata but forbidden in feature payloads.
    """

    metadata = episode_features.get("metadata", {})
    if "learning_type" in episode_features.get("episode_features", {}):
        raise ValueError("learning_type must not be present in episode_features")

    n_layers = int(
        config.get("n_layers")
        or metadata.get("n_layers")
        or episode_features.get("n_layers")
        or 32
    )
    module_rows = list(episode_features.get("module_features", []))
    selected_layers = selected_layers_for_config(config, n_layers=n_layers)
    target_modules = set(config.get("target_modules") or [])

    selected: List[Dict[str, Any]] = []
    outside: List[Dict[str, Any]] = []
    for module_row in module_rows:
        if module_is_selected(module_row, selected_layers, target_modules):
            selected.append(module_row)
        else:
            outside.append(module_row)
    total_modules = len(module_rows)
    validate_module_selection(config, module_rows, selected, target_modules)

    named: List[Tuple[str, float]] = []
    named.extend(
        numeric_episode_features(
            episode_features.get("episode_features", episode_features),
            feature_set=feature_set,
        )
    )
    if include_group(feature_set, "module_probes"):
        named.extend(aggregate_module_rows("selected", selected))
        named.extend(aggregate_module_rows("outside", outside))

    module_family = str(config.get("module_family", "custom"))
    for family in MODULE_FAMILIES:
        named.append((f"config__module_family__{family}", 1.0 if module_family == family else 0.0))

    layer_summary = layer_summary_for_config(config, n_layers=n_layers)
    named.extend(
        [
            ("config__is_full_stack", layer_summary["is_full_stack"]),
            ("config__is_localized", layer_summary["is_localized"]),
            ("config__n_layers_adapted", layer_summary["n_layers_adapted"]),
            ("config__fraction_layers_adapted", layer_summary["fraction_layers_adapted"]),
            ("config__depth_mean", layer_summary["depth_mean"]),
            ("config__depth_std", layer_summary["depth_std"]),
            ("config__depth_min", layer_summary["depth_min"]),
            ("config__depth_max", layer_summary["depth_max"]),
            ("config__fraction_modules_selected", len(selected) / total_modules if total_modules else 0.0),
            ("config__lora_r", float(config.get("lora_r", 0) or 0)),
            ("config__lora_alpha", float(config.get("lora_alpha", 0) or 0)),
            ("config__lora_dropout", float(config.get("lora_dropout", 0.0) or 0.0)),
            (
                "config__approximate_parameter_cost",
                float(config.get("approximate_parameter_cost") or approximate_parameter_cost(config, n_layers=n_layers)),
            ),
        ]
    )

    names = [name for name, _ in named]
    values = [float(value) for _, value in named]
    return values, names
