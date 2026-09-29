#!/usr/bin/env python3
"""Backbone metadata for adaptation-compiler replications.

The compiler pipeline is intentionally model-parameterized. This module keeps
the small amount of backbone-specific information needed by orchestration code
out of the Llama protocol and away from scoring/training logic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

try:  # pragma: no cover
    from .compiler_common import DEFAULT_TARGET_MODULES, model_slug
    from .make_compiler_config_library import build_config_library
except ImportError:  # pragma: no cover
    from compiler_common import DEFAULT_TARGET_MODULES, model_slug
    from make_compiler_config_library import build_config_library


PRIMARY_COMPILER_CONFIG_IDS = [
    "early__all__r16",
    "middle__all__r16",
    "late__all__r16",
    "full__all__r4",
]

HIGH_CAPACITY_CONFIG_ID = "full__all__r16"


@dataclass(frozen=True)
class BackboneSpec:
    model_name: str
    n_layers: int
    target_modules: tuple[str, ...] = tuple(DEFAULT_TARGET_MODULES)
    layers_pattern: str = "layers"
    torch_dtype: str = "bfloat16"
    device_map: str = "auto"
    normalized_window_fraction: float = 0.25

    @property
    def slug(self) -> str:
        return model_slug(self.model_name)

    @property
    def region_width(self) -> int:
        return normalized_region_width(self.n_layers, self.normalized_window_fraction)


KNOWN_BACKBONES: Dict[str, BackboneSpec] = {
    "meta-llama/Llama-3.1-8B-Instruct": BackboneSpec(
        model_name="meta-llama/Llama-3.1-8B-Instruct",
        n_layers=32,
    ),
    "google/gemma-2-9b-it": BackboneSpec(
        model_name="google/gemma-2-9b-it",
        n_layers=42,
    ),
}


def normalized_region_width(n_layers: int, fraction: float = 0.25) -> int:
    """Return a contiguous localization width based on normalized depth.

    Llama-3.1-8B has 32 layers, so 25 percent gives width 8. Gemma-2-9B has
    42 layers, so the same normalized window rounds up to width 11.
    """

    if n_layers <= 0:
        raise ValueError("n_layers must be positive.")
    if fraction <= 0:
        raise ValueError("fraction must be positive.")
    return max(1, int(math.ceil(float(n_layers) * float(fraction))))


def resolve_backbone_spec(
    model_name: str,
    *,
    n_layers: int | None = None,
    target_modules: Sequence[str] | None = None,
    layers_pattern: str | None = None,
    torch_dtype: str | None = None,
    device_map: str | None = None,
) -> BackboneSpec:
    base = KNOWN_BACKBONES.get(model_name)
    if base is None and n_layers is None:
        raise ValueError(
            f"Unknown backbone {model_name!r}. Pass --n_layers to use it in compiler orchestration."
        )
    if base is None:
        base = BackboneSpec(model_name=model_name, n_layers=int(n_layers))
    return BackboneSpec(
        model_name=model_name,
        n_layers=int(n_layers if n_layers is not None else base.n_layers),
        target_modules=tuple(target_modules if target_modules is not None else base.target_modules),
        layers_pattern=layers_pattern if layers_pattern is not None else base.layers_pattern,
        torch_dtype=torch_dtype if torch_dtype is not None else base.torch_dtype,
        device_map=device_map if device_map is not None else base.device_map,
        normalized_window_fraction=base.normalized_window_fraction,
    )


def build_backbone_config_library(
    spec: BackboneSpec,
    *,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    include_capacity_matched_full: bool = True,
) -> List[Dict[str, Any]]:
    """Build the compiler program library for one backbone.

    The returned rows use the same schema as ``data/compiler/config_library.jsonl``.
    For Gemma this yields early/middle/late width-11 localized rank-16 programs,
    the high-capacity full-r16 baseline, and the capacity-matched full-r4
    candidate used in the primary compiler comparison.
    """

    return build_config_library(
        target_modules=spec.target_modules,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        n_layers=spec.n_layers,
        region_width=spec.region_width,
        layers_pattern=spec.layers_pattern,
        include_capacity_matched_full=include_capacity_matched_full,
    )


def discover_semantic_target_modules(
    module_names: Sequence[str],
    target_modules: Sequence[str] = DEFAULT_TARGET_MODULES,
) -> Dict[str, Any]:
    """Report which semantic LoRA target modules are present by name suffix.

    This is intentionally a lightweight validation helper. The actual trainer
    and PEFT still perform the authoritative attachment when the model is loaded.
    """

    present = sorted(
        {
            target
            for name in module_names
            for target in target_modules
            if str(name).endswith(f".{target}") or str(name) == target
        }
    )
    missing = [target for target in target_modules if target not in present]
    return {
        "target_modules": list(target_modules),
        "present": present,
        "missing": missing,
        "all_present": not missing,
    }
