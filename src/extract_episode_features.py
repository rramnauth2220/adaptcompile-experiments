#!/usr/bin/env python3
"""Extract pre-adaptation episode features for compiler prediction.

The CLI defaults to a real frozen-backbone backend. It computes supervised
target-token negative log-likelihood and memory-safe module-output gradient
statistics without adapting the model. A deterministic proxy backend remains
available for tests and dry infrastructure via ``--backend proxy``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:  # pragma: no cover
    from .common import format_prompt
    from .compiler_common import (
        DEFAULT_TARGET_MODULES,
        model_slug,
        read_jsonl,
    )
except ImportError:  # pragma: no cover
    from common import format_prompt
    from compiler_common import DEFAULT_TARGET_MODULES, model_slug, read_jsonl


def stable_unit_interval(*parts: Any) -> float:
    payload = "||".join(str(p) for p in parts).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    return int(digest[:12], 16) / float(16**12 - 1)


def stats(values: Sequence[float]) -> Dict[str, float]:
    vals = [float(v) for v in values]
    if not vals:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    mean = sum(vals) / len(vals)
    var = sum((v - mean) ** 2 for v in vals) / len(vals)
    return {
        "mean": mean,
        "std": math.sqrt(var),
        "min": min(vals),
        "max": max(vals),
    }


def pseudo_token_count(text: str) -> int:
    return max(1, len(str(text).strip().split()))


def loss_proxy_for_example(example: Dict[str, Any]) -> float:
    """A deterministic frozen-loss placeholder based on target/prompt shape.

    This is not an exact model loss. It is a cheap schema-compatible diagnostic
    used for tests and dry infrastructure until a frozen model is supplied.
    """

    prompt_len = pseudo_token_count(format_prompt(example["prompt"]))
    target_len = pseudo_token_count(example["target"])
    content_hash = stable_unit_interval(example.get("prompt", ""), example.get("target", ""))
    return math.log1p(target_len) + 0.01 * math.log1p(prompt_len) + 0.05 * content_hash


def content_proxy_embedding_for_example(
    example: Dict[str, Any],
    projection_dim: int,
    seed: int,
) -> List[float]:
    """Deterministic content-only representation used by proxy tests."""

    if projection_dim <= 0:
        return []
    prompt = format_prompt(example["prompt"])
    target = str(example["target"])
    return [
        2.0 * stable_unit_interval(seed, "episode_embedding", j, prompt, target) - 1.0
        for j in range(projection_dim)
    ]


def mean_vectors(vectors: Sequence[Sequence[float]], dim: int) -> List[float]:
    if dim <= 0:
        return []
    if not vectors:
        return [0.0] * dim
    return [
        sum(float(vec[j]) for vec in vectors) / len(vectors)
        for j in range(dim)
    ]


def embedding_feature_payload(values: Sequence[float]) -> Dict[str, float]:
    return {
        f"episode_embedding_{idx:03d}": float(value)
        for idx, value in enumerate(values)
    }


def module_type_weight(module_type: str) -> float:
    if module_type in {"q_proj", "k_proj", "v_proj", "o_proj"}:
        return 1.0
    if module_type in {"gate_proj", "up_proj", "down_proj"}:
        return 1.15
    return 0.9


def build_module_features(
    episode_id: str,
    loss_values: Sequence[float],
    target_modules: Sequence[str],
    n_layers: int,
) -> List[Dict[str, Any]]:
    loss_stats = stats(loss_values)
    rows: List[Dict[str, Any]] = []
    for layer in range(n_layers):
        normalized_depth = layer / max(1, n_layers - 1)
        for module_type in target_modules:
            deterministic_jitter = stable_unit_interval(episode_id, layer, module_type) - 0.5
            sensitivity = (
                loss_stats["mean"]
                * module_type_weight(module_type)
                * (0.75 + 0.5 * normalized_depth)
                * (1.0 + 0.05 * deterministic_jitter)
            )
            gradient_magnitude = sensitivity / (1.0 + normalized_depth)
            activation_rms = math.sqrt(max(loss_stats["mean"], 0.0) + 1e-8) * (1.0 + 0.1 * normalized_depth)
            agreement = 1.0 / (1.0 + loss_stats["std"] + abs(deterministic_jitter) * 0.05)
            rows.append(
                {
                    "layer_index": layer,
                    "normalized_depth": normalized_depth,
                    "module_type": module_type,
                    "sensitivity_proxy": sensitivity,
                    "sensitivity_mean": sensitivity,
                    "gradient_magnitude_mean": gradient_magnitude,
                    "gradient_magnitude_std": 0.0,
                    "activation_rms_mean": activation_rms,
                    "activation_rms_std": 0.0,
                    "gradient_agreement_proxy": agreement,
                    "probe_backend": "loss_proxy",
                }
            )
    return rows


def infer_num_layers_from_config(config: Any) -> int:
    for attr in ["num_hidden_layers", "n_layer", "num_layers"]:
        if hasattr(config, attr):
            return int(getattr(config, attr))
    raise ValueError("Could not infer number of transformer layers from model config.")


def dtype_from_arg(torch_module: Any, arg: str) -> Any:
    if arg == "auto":
        return "auto"
    return {
        "float16": torch_module.float16,
        "bfloat16": torch_module.bfloat16,
        "float32": torch_module.float32,
    }[arg]


def require_model_deps() -> Tuple[Any, Any, Any]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "The frozen_model backend requires torch and transformers. "
            "Use --backend proxy only for fixture tests."
        ) from exc
    return torch, AutoModelForCausalLM, AutoTokenizer


def input_device_for(model: Any, torch_module: Any) -> Any:
    device = getattr(model, "device", None)
    if device is not None:
        return device
    for param in model.parameters():
        if getattr(param, "device", None) is not None and param.device.type != "meta":
            return param.device
    return torch_module.device("cpu")


def encode_supervised_example(
    example: Dict[str, Any],
    tokenizer: Any,
    max_length: int,
    torch_module: Any,
    device: Any,
) -> Dict[str, Any]:
    prompt = format_prompt(example["prompt"])
    eos = tokenizer.eos_token or ""
    answer = example["target"].strip() + eos
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]
    input_ids = (prompt_ids + answer_ids)[:max_length]
    labels = ([-100] * len(prompt_ids) + answer_ids)[:max_length]
    if not any(label != -100 for label in labels):
        raise ValueError(f"No target tokens remain after truncation for {example.get('example_id')}")
    attention_mask = [1] * len(input_ids)
    return {
        "input_ids": torch_module.tensor([input_ids], dtype=torch_module.long, device=device),
        "attention_mask": torch_module.tensor([attention_mask], dtype=torch_module.long, device=device),
        "labels": torch_module.tensor([labels], dtype=torch_module.long, device=device),
        "n_target_tokens": sum(1 for label in labels if label != -100),
    }


def target_token_loss(model: Any, batch: Dict[str, Any], torch_module: Any) -> Any:
    outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    logits = outputs.logits
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = batch["labels"][:, 1:].contiguous()
    mask = shift_labels.ne(-100)
    if not bool(mask.any()):
        raise ValueError("No target labels available for loss computation.")
    active_logits = shift_logits[mask].float()
    active_labels = shift_labels[mask]
    losses = torch_module.nn.functional.cross_entropy(
        active_logits,
        active_labels,
        reduction="none",
    )
    return losses.mean()


def target_token_loss_and_hidden_mean(
    model: Any,
    batch: Dict[str, Any],
    torch_module: Any,
    hidden_state_layer: int,
) -> Tuple[Any, Any]:
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        output_hidden_states=True,
        use_cache=False,
    )
    logits = outputs.logits
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = batch["labels"][:, 1:].contiguous()
    loss_mask = shift_labels.ne(-100)
    if not bool(loss_mask.any()):
        raise ValueError("No target labels available for loss computation.")
    losses = torch_module.nn.functional.cross_entropy(
        shift_logits[loss_mask].float(),
        shift_labels[loss_mask],
        reduction="none",
    )

    hidden_states = outputs.hidden_states
    if hidden_states is None:
        raise ValueError("Model did not return hidden states for episode representation.")
    hidden = hidden_states[int(hidden_state_layer)]
    hidden_mask = batch["labels"].ne(-100)
    if not bool(hidden_mask.any()):
        raise ValueError("No target tokens available for hidden-state pooling.")
    pooled = hidden[hidden_mask].float().mean(dim=0)
    return losses.mean(), pooled


def parse_layer_and_module(name: str, target_modules: Sequence[str]) -> Optional[Tuple[int, str]]:
    parts = name.split(".")
    module_type = parts[-1]
    if module_type not in target_modules:
        return None
    for marker in ["layers", "h", "blocks"]:
        for idx, part in enumerate(parts[:-1]):
            if part == marker and idx + 1 < len(parts) and parts[idx + 1].isdigit():
                return int(parts[idx + 1]), module_type
    match = re.search(r"\.(\d+)\.[^.]+$", name)
    if match:
        return int(match.group(1)), module_type
    return None


def discover_probe_modules(
    model: Any,
    target_modules: Sequence[str],
    n_layers: int,
) -> List[Dict[str, Any]]:
    probes: List[Dict[str, Any]] = []
    for name, module in model.named_modules():
        parsed = parse_layer_and_module(name, target_modules)
        if parsed is None:
            continue
        layer_index, module_type = parsed
        if 0 <= layer_index < n_layers:
            probes.append(
                {
                    "module_key": f"{layer_index:04d}::{module_type}::{name}",
                    "name": name,
                    "module": module,
                    "layer_index": layer_index,
                    "module_type": module_type,
                    "normalized_depth": layer_index / max(1, n_layers - 1),
                }
            )
    if not probes:
        raise ValueError(
            "Could not find probe modules matching target_modules="
            f"{list(target_modules)}. Override --target_modules for this model family."
        )
    return sorted(probes, key=lambda row: (row["layer_index"], row["module_type"], row["name"]))


def stable_int(*parts: Any) -> int:
    payload = "||".join(str(p) for p in parts).encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:8], 16)


def gradient_sketch_seed(
    extractor_seed: int,
    episode_id: str,
    module_key: str,
    example_idx: int | None = None,
) -> int:
    """Seed the gradient sketch basis shared across examples for a module.

    ``example_idx`` is accepted only for regression tests and old call sites;
    it is intentionally ignored. Cross-example gradient agreement is meaningful
    only when all examples for the same episode/module use the same projection
    basis.
    """

    return stable_int(extractor_seed, episode_id, module_key)


def sketch_tensor(
    tensor: Any,
    torch_module: Any,
    sketch_dim: int,
    max_elements: int,
    seed: int,
) -> List[float]:
    flat = tensor.detach().float().flatten()
    n = int(flat.numel())
    if n == 0 or sketch_dim <= 0:
        return [0.0] * max(1, sketch_dim)
    k = min(n, max_elements)
    if k == n:
        idx = torch_module.arange(n, device=flat.device, dtype=torch_module.long)
    else:
        idx = torch_module.linspace(0, n - 1, steps=k, device=flat.device).long()
    vals = flat[idx]
    seed_mod = int(seed) % 2_147_483_647
    bins = ((idx * 1_103_515_245 + seed_mod) % sketch_dim).long()
    signs = (((idx * 1_664_525 + seed_mod) % 2).float() * 2.0) - 1.0
    sketch = torch_module.zeros(sketch_dim, device=flat.device, dtype=torch_module.float32)
    sketch.scatter_add_(0, bins, vals * signs)
    norm = torch_module.linalg.vector_norm(sketch)
    if float(norm.item()) > 0:
        sketch = sketch / norm
    return sketch.detach().cpu().tolist()


def mean_pairwise_cosine(sketches: Sequence[Sequence[float]]) -> float:
    if len(sketches) < 2:
        return 1.0
    total = 0.0
    count = 0
    for i in range(len(sketches)):
        for j in range(i + 1, len(sketches)):
            total += sum(float(a) * float(b) for a, b in zip(sketches[i], sketches[j]))
            count += 1
    return total / count if count else 1.0


def chunks(rows: Sequence[Dict[str, Any]], size: int) -> Iterable[List[Dict[str, Any]]]:
    size = max(1, int(size))
    for start in range(0, len(rows), size):
        yield list(rows[start : start + size])


def resolve_probe_module_batch_size(requested: int, n_probes: int) -> int:
    if n_probes <= 0:
        return 1
    if int(requested) < 0:
        return int(n_probes)
    return max(1, min(int(requested), int(n_probes)))


def theoretical_probe_passes(n_probe_examples: int, n_probe_modules: int, probe_module_batch_size: int) -> int:
    if n_probe_examples <= 0 or n_probe_modules <= 0:
        return 0
    batch_size = resolve_probe_module_batch_size(probe_module_batch_size, n_probe_modules)
    return int(n_probe_examples) * math.ceil(int(n_probe_modules) / batch_size)


def make_activation_hook(
    module_key: str,
    activations: Dict[str, Any],
    torch_module: Any,
):
    def hook(_module: Any, _inputs: Tuple[Any, ...], output: Any) -> Any:
        if torch_module.is_tensor(output):
            if not output.requires_grad:
                output.requires_grad_(True)
            output.retain_grad()
            activations[module_key] = output
            return output
        if isinstance(output, tuple) and output and torch_module.is_tensor(output[0]):
            first = output[0]
            if not first.requires_grad:
                first.requires_grad_(True)
            first.retain_grad()
            activations[module_key] = first
            return output
        return output

    return hook


def update_probe_record(
    record: Dict[str, Any],
    activation: Any,
    gradient: Any,
    torch_module: Any,
    sketch_dim: int,
    sketch_elements: int,
    seed: int,
) -> None:
    with torch_module.no_grad():
        act = activation.detach().float()
        grad = gradient.detach().float()
        sensitivity = torch_module.mean(torch_module.abs(act * grad)).item()
        grad_rms = torch_module.sqrt(torch_module.mean(grad * grad)).item()
        act_rms = torch_module.sqrt(torch_module.mean(act * act)).item()
        record.setdefault("_sensitivity_values", []).append(float(sensitivity))
        record.setdefault("_gradient_rms_values", []).append(float(grad_rms))
        record.setdefault("_activation_rms_values", []).append(float(act_rms))
        record.setdefault("_gradient_sketches", []).append(
            sketch_tensor(
                grad,
                torch_module=torch_module,
                sketch_dim=sketch_dim,
                max_elements=sketch_elements,
                seed=seed,
            )
        )


def finalize_probe_records(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for record in records:
        sens = record.pop("_sensitivity_values", [])
        grad = record.pop("_gradient_rms_values", [])
        act = record.pop("_activation_rms_values", [])
        sketches = record.pop("_gradient_sketches", [])
        sens_stats = stats(sens)
        grad_stats = stats(grad)
        act_stats = stats(act)
        out.append(
            {
                **record,
                "sensitivity_proxy": sens_stats["mean"],
                "sensitivity_mean": sens_stats["mean"],
                "sensitivity_std": sens_stats["std"],
                "gradient_magnitude_mean": grad_stats["mean"],
                "gradient_magnitude_std": grad_stats["std"],
                "activation_rms_mean": act_stats["mean"],
                "activation_rms_std": act_stats["std"],
                "gradient_agreement_proxy": mean_pairwise_cosine(sketches),
                "n_probe_examples": len(sens),
                "probe_backend": "frozen_model_module_output",
                "gradient_agreement_note": (
                    "Cosine similarity over deterministic compact sketches of "
                    "module-output gradients, not exact parameter-gradient cosine."
                ),
            }
        )
    return out


def clear_cuda_cache(torch_module: Any) -> None:
    if hasattr(torch_module, "cuda") and torch_module.cuda.is_available():
        torch_module.cuda.empty_cache()


class FrozenModelFeatureBackend:
    """Reusable frozen-backbone feature backend for many episodes.

    Loading an 8B+ checkpoint dominates feature-extraction wall time. The CLI
    instantiates this backend once per command and calls ``extract`` for each
    selected episode, so batched episode extraction no longer reloads the model.
    """

    def __init__(
        self,
        model_name_or_path: str,
        target_modules: Sequence[str],
        n_layers: Optional[int],
        seed: int,
        max_length: int,
        torch_dtype: str,
        device_map: str,
        max_probe_examples: Optional[int],
        probe_module_batch_size: int,
        sketch_dim: int,
        sketch_elements: int,
        episode_embedding_dim: int,
        episode_embedding_layer: int,
        benchmark: bool,
        trust_remote_code: bool,
    ) -> None:
        torch_module, AutoModelForCausalLM, AutoTokenizer = require_model_deps()
        torch_module.manual_seed(seed)

        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            use_fast=True,
            trust_remote_code=trust_remote_code,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            device_map=device_map,
            torch_dtype=dtype_from_arg(torch_module, torch_dtype),
            trust_remote_code=trust_remote_code,
        )
        model.eval()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False
        for param in model.parameters():
            param.requires_grad_(False)

        inferred_layers = infer_num_layers_from_config(model.config)
        self.torch_module = torch_module
        self.tokenizer = tokenizer
        self.model = model
        self.model_name_or_path = model_name_or_path
        self.target_modules = list(target_modules)
        self.seed = seed
        self.max_length = max_length
        self.torch_dtype = torch_dtype
        self.device_map = device_map
        self.max_probe_examples = max_probe_examples
        self.requested_probe_module_batch_size = int(probe_module_batch_size)
        self.probe_module_batch_size = int(probe_module_batch_size)
        self.sketch_dim = sketch_dim
        self.sketch_elements = sketch_elements
        self.episode_embedding_dim = int(episode_embedding_dim)
        self.episode_embedding_layer = int(episode_embedding_layer)
        self.benchmark = bool(benchmark)
        self.episode_embedding_projection_seed = stable_int(
            seed,
            model_name_or_path,
            "episode_embedding_projection",
            self.episode_embedding_layer,
            self.episode_embedding_dim,
        )
        self._episode_projection_cache: Dict[int, Any] = {}
        self.inferred_layers = int(inferred_layers)
        self.n_layers = int(n_layers or inferred_layers)
        self.device = input_device_for(model, torch_module)
        self.probes = discover_probe_modules(
            model,
            target_modules=self.target_modules,
            n_layers=self.n_layers,
        )
        self.probe_module_batch_size = resolve_probe_module_batch_size(
            self.requested_probe_module_batch_size,
            n_probes=len(self.probes),
        )

    def project_episode_hidden(self, hidden_mean: Any) -> List[float]:
        if self.episode_embedding_dim <= 0:
            return []
        torch_module = self.torch_module
        vector = hidden_mean.detach().float().cpu().flatten()
        input_dim = int(vector.numel())
        if input_dim <= 0:
            return [0.0] * self.episode_embedding_dim
        if input_dim not in self._episode_projection_cache:
            generator = torch_module.Generator(device="cpu")
            generator.manual_seed(int(self.episode_embedding_projection_seed) % (2**63 - 1))
            projection = torch_module.empty(
                (input_dim, self.episode_embedding_dim),
                dtype=torch_module.float32,
                device="cpu",
            )
            projection.normal_(mean=0.0, std=1.0 / math.sqrt(input_dim), generator=generator)
            self._episode_projection_cache[input_dim] = projection
        projected = vector @ self._episode_projection_cache[input_dim]
        return projected.detach().cpu().tolist()

    def encode_examples(self, train_examples: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [
            encode_supervised_example(
                example,
                tokenizer=self.tokenizer,
                max_length=self.max_length,
                torch_module=self.torch_module,
                device=self.device,
            )
            for example in train_examples
        ]

    def fresh_probe_records(self) -> Dict[str, Dict[str, Any]]:
        return {
            probe["module_key"]: {
                "layer_index": probe["layer_index"],
                "normalized_depth": probe["normalized_depth"],
                "module_type": probe["module_type"],
                "module_name": probe["name"],
            }
            for probe in self.probes
        }

    def extract(
        self,
        episode: Dict[str, Any],
        examples_by_id: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        torch_module = self.torch_module
        train_examples = [examples_by_id[eid] for eid in episode["train_example_ids"]]
        encoded = self.encode_examples(train_examples)

        loss_values: List[float] = []
        target_token_counts: List[int] = []
        projected_embeddings: List[List[float]] = []
        with torch_module.no_grad():
            for batch in encoded:
                loss, hidden_mean = target_token_loss_and_hidden_mean(
                    self.model,
                    batch,
                    torch_module,
                    hidden_state_layer=self.episode_embedding_layer,
                )
                loss_values.append(float(loss.detach().cpu().item()))
                target_token_counts.append(int(batch["n_target_tokens"]))
                projected_embeddings.append(self.project_episode_hidden(hidden_mean))

        probe_records = self.fresh_probe_records()
        n_probe = (
            len(encoded)
            if self.max_probe_examples is None
            else min(len(encoded), self.max_probe_examples)
        )
        probe_passes = 0
        probe_start = time.perf_counter()
        peak_cuda_memory = None
        if hasattr(torch_module, "cuda") and torch_module.cuda.is_available():
            try:
                torch_module.cuda.reset_peak_memory_stats()
            except Exception:
                pass
        for example_idx, batch in enumerate(encoded[:n_probe]):
            for probe_batch in chunks(self.probes, self.probe_module_batch_size):
                activations: Dict[str, Any] = {}
                handles = [
                    probe["module"].register_forward_hook(
                        make_activation_hook(probe["module_key"], activations, torch_module)
                    )
                    for probe in probe_batch
                ]
                try:
                    self.model.zero_grad(set_to_none=True)
                    loss = target_token_loss(self.model, batch, torch_module)
                    loss.backward()
                    probe_passes += 1
                    for probe in probe_batch:
                        activation = activations.get(probe["module_key"])
                        if activation is None or activation.grad is None:
                            continue
                        update_probe_record(
                            probe_records[probe["module_key"]],
                            activation=activation,
                            gradient=activation.grad,
                            torch_module=torch_module,
                            sketch_dim=self.sketch_dim,
                            sketch_elements=self.sketch_elements,
                            seed=gradient_sketch_seed(
                                self.seed,
                                episode["episode_id"],
                                probe["module_key"],
                                example_idx=example_idx,
                            ),
                        )
                except RuntimeError as exc:
                    message = str(exc).lower()
                    if "out of memory" in message or "cuda oom" in message:
                        raise RuntimeError(
                            "CUDA out of memory during module-output probe batching. "
                            "Reduce --probe_module_batch_size, for example try 1, 8, or 32."
                        ) from exc
                    raise
                finally:
                    for handle in handles:
                        handle.remove()
                    activations.clear()
                    self.model.zero_grad(set_to_none=True)
                    clear_cuda_cache(torch_module)
        probe_elapsed = time.perf_counter() - probe_start
        if hasattr(torch_module, "cuda") and torch_module.cuda.is_available():
            try:
                peak_cuda_memory = int(torch_module.cuda.max_memory_allocated())
            except Exception:
                peak_cuda_memory = None
        expected_probe_passes = theoretical_probe_passes(
            n_probe_examples=n_probe,
            n_probe_modules=len(self.probes),
            probe_module_batch_size=self.probe_module_batch_size,
        )
        if self.benchmark:
            print(
                "[benchmark] "
                f"episode_id={episode['episode_id']} "
                f"n_probe_modules={len(self.probes)} "
                f"n_probe_examples={n_probe} "
                f"probe_module_batch_size={self.probe_module_batch_size} "
                f"probe_forward_backward_passes={probe_passes} "
                f"theoretical_probe_passes={expected_probe_passes} "
                f"probe_elapsed_seconds={probe_elapsed:.3f} "
                f"peak_cuda_memory_bytes={peak_cuda_memory}"
            )

        loss_stats = stats(loss_values)
        token_stats = stats(target_token_counts)
        episode_feature_payload = {
            "n_adaptation_examples": len(train_examples),
            "frozen_target_token_loss_mean": loss_stats["mean"],
            "frozen_target_token_loss_std": loss_stats["std"],
            "frozen_target_token_loss_min": loss_stats["min"],
            "frozen_target_token_loss_max": loss_stats["max"],
            "target_token_count_mean": token_stats["mean"],
            "target_token_count_std": token_stats["std"],
            "target_token_count_min": token_stats["min"],
            "target_token_count_max": token_stats["max"],
        }
        episode_feature_payload.update(
            embedding_feature_payload(
                mean_vectors(projected_embeddings, dim=self.episode_embedding_dim)
            )
        )

        return {
            "schema_version": 1,
            "metadata": {
                "episode_id": episode["episode_id"],
                "spec_ids": episode["spec_ids"],
                "meta_split": episode["meta_split"],
                "learning_type": episode["learning_type"],
                "model_name": self.model_name_or_path,
                "model_slug": model_slug(self.model_name_or_path),
                "seed": self.seed,
                "n_layers": int(self.n_layers),
                "inferred_n_layers": int(self.inferred_layers),
                "target_modules": list(self.target_modules),
                "probe_backend": "frozen_model",
                "probe_backend_note": (
                    "Actual frozen-backbone target-token NLL plus module-output "
                    "gradient sensitivity. Gradient agreement uses compact sketches "
                    "of module-output gradients; it is not exact parameter-gradient agreement."
                ),
                "max_length": self.max_length,
                "torch_dtype": self.torch_dtype,
                "device_map": self.device_map,
                "max_probe_examples": n_probe,
                "requested_probe_module_batch_size": self.requested_probe_module_batch_size,
                "probe_module_batch_size": self.probe_module_batch_size,
                "probe_module_batching_note": (
                    "Graph-preserving hooks retain gradients on original module outputs; "
                    "-1 resolves to all discovered probe modules."
                ),
                "probe_forward_backward_passes": probe_passes,
                "theoretical_probe_forward_backward_passes": expected_probe_passes,
                "probe_elapsed_seconds": probe_elapsed,
                "peak_cuda_memory_bytes": peak_cuda_memory,
                "sketch_dim": self.sketch_dim,
                "sketch_elements": self.sketch_elements,
                "n_probe_modules": len(self.probes),
                "episode_representation_backend": "frozen_hidden_state_projection",
                "episode_representation_source_layer": self.episode_embedding_layer,
                "episode_representation_token_pooling": "mean over target tokens",
                "episode_representation_episode_pooling": "mean over adaptation examples",
                "episode_representation_projection_dim": self.episode_embedding_dim,
                "episode_representation_projection_seed": self.episode_embedding_projection_seed,
            },
            "episode_features": episode_feature_payload,
            "module_features": finalize_probe_records(list(probe_records.values())),
        }


def compute_frozen_model_features(
    episode: Dict[str, Any],
    examples_by_id: Dict[str, Dict[str, Any]],
    model_name_or_path: str,
    target_modules: Sequence[str],
    n_layers: Optional[int],
    seed: int,
    max_length: int,
    torch_dtype: str,
    device_map: str,
    max_probe_examples: Optional[int],
    probe_module_batch_size: int,
    sketch_dim: int,
    sketch_elements: int,
    episode_embedding_dim: int,
    episode_embedding_layer: int,
    benchmark: bool,
    trust_remote_code: bool,
) -> Dict[str, Any]:
    backend = FrozenModelFeatureBackend(
        model_name_or_path=model_name_or_path,
        target_modules=target_modules,
        n_layers=n_layers,
        seed=seed,
        max_length=max_length,
        torch_dtype=torch_dtype,
        device_map=device_map,
        max_probe_examples=max_probe_examples,
        probe_module_batch_size=probe_module_batch_size,
        sketch_dim=sketch_dim,
        sketch_elements=sketch_elements,
        episode_embedding_dim=episode_embedding_dim,
        episode_embedding_layer=episode_embedding_layer,
        benchmark=benchmark,
        trust_remote_code=trust_remote_code,
    )
    return backend.extract(episode, examples_by_id=examples_by_id)


def compute_proxy_features(
    episode: Dict[str, Any],
    examples_by_id: Dict[str, Dict[str, Any]],
    model_name_or_path: str,
    target_modules: Sequence[str],
    n_layers: int,
    seed: int,
    episode_embedding_dim: int = 128,
) -> Dict[str, Any]:
    train_examples = [examples_by_id[eid] for eid in episode["train_example_ids"]]
    loss_values = [loss_proxy_for_example(example) for example in train_examples]
    loss_stats = stats(loss_values)
    projected_embeddings = [
        content_proxy_embedding_for_example(
            example,
            projection_dim=episode_embedding_dim,
            seed=seed,
        )
        for example in train_examples
    ]

    episode_feature_payload = {
        "n_adaptation_examples": len(train_examples),
        "frozen_target_token_loss_mean": loss_stats["mean"],
        "frozen_target_token_loss_std": loss_stats["std"],
        "frozen_target_token_loss_min": loss_stats["min"],
        "frozen_target_token_loss_max": loss_stats["max"],
    }
    episode_feature_payload.update(
        embedding_feature_payload(mean_vectors(projected_embeddings, dim=episode_embedding_dim))
    )

    # Important: learning_type is metadata only, not a predictor input.
    return {
        "schema_version": 1,
        "metadata": {
            "episode_id": episode["episode_id"],
            "spec_ids": episode["spec_ids"],
            "meta_split": episode["meta_split"],
            "learning_type": episode["learning_type"],
            "model_name": model_name_or_path,
            "model_slug": model_slug(model_name_or_path),
            "seed": seed,
            "n_layers": int(n_layers),
            "target_modules": list(target_modules),
            "probe_backend": "proxy",
            "probe_backend_note": (
                "Deterministic memory-light proxy, not exact frozen-model parameter gradients."
            ),
            "episode_representation_backend": "proxy_content_hash",
            "episode_representation_source_layer": None,
            "episode_representation_token_pooling": "content hash over prompt and target",
            "episode_representation_episode_pooling": "mean over adaptation examples",
            "episode_representation_projection_dim": int(episode_embedding_dim),
            "episode_representation_projection_seed": seed,
        },
        "episode_features": episode_feature_payload,
        "module_features": build_module_features(
            episode_id=episode["episode_id"],
            loss_values=loss_values,
            target_modules=target_modules,
            n_layers=n_layers,
        ),
    }


def extract_episode_features(
    episode: Dict[str, Any],
    examples_by_id: Dict[str, Dict[str, Any]],
    model_name_or_path: str,
    target_modules: Sequence[str] = DEFAULT_TARGET_MODULES,
    n_layers: Optional[int] = None,
    seed: int = 2026,
    backend: str = "frozen_model",
    max_length: int = 512,
    torch_dtype: str = "auto",
    device_map: str = "auto",
    max_probe_examples: Optional[int] = 4,
    probe_module_batch_size: int = 1,
    sketch_dim: int = 64,
    sketch_elements: int = 4096,
    episode_embedding_dim: int = 128,
    episode_embedding_layer: int = -1,
    benchmark: bool = False,
    trust_remote_code: bool = False,
) -> Dict[str, Any]:
    if backend == "proxy":
        return compute_proxy_features(
            episode,
            examples_by_id=examples_by_id,
            model_name_or_path=model_name_or_path,
            target_modules=target_modules,
            n_layers=int(n_layers or 32),
            seed=seed,
            episode_embedding_dim=episode_embedding_dim,
        )
    if backend != "frozen_model":
        raise ValueError(f"Unknown feature backend: {backend}")
    return compute_frozen_model_features(
        episode,
        examples_by_id=examples_by_id,
        model_name_or_path=model_name_or_path,
        target_modules=target_modules,
        n_layers=n_layers,
        seed=seed,
        max_length=max_length,
        torch_dtype=torch_dtype,
        device_map=device_map,
        max_probe_examples=max_probe_examples,
        probe_module_batch_size=probe_module_batch_size,
        sketch_dim=sketch_dim,
        sketch_elements=sketch_elements,
        episode_embedding_dim=episode_embedding_dim,
        episode_embedding_layer=episode_embedding_layer,
        benchmark=benchmark,
        trust_remote_code=trust_remote_code,
    )


def select_episodes(
    episodes: Sequence[Dict[str, Any]],
    episode_ids: Sequence[str] | None,
    meta_split: str | None,
    max_episodes: int | None,
) -> List[Dict[str, Any]]:
    selected = list(episodes)
    if episode_ids:
        allowed = set(episode_ids)
        selected = [row for row in selected if row["episode_id"] in allowed]
    if meta_split:
        selected = [row for row in selected if row["meta_split"] == meta_split]
    selected = sorted(selected, key=lambda row: row["episode_id"])
    if max_episodes is not None:
        selected = selected[:max_episodes]
    return selected


def feature_path(output_root: Path, model_name_or_path: str, episode_id: str) -> Path:
    safe_episode = episode_id.replace("::", "__").replace("/", "_")
    return output_root / model_slug(model_name_or_path) / f"{safe_episode}.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract compiler episode features.")
    parser.add_argument("--episode_manifest", default="data/compiler/episode_manifest.jsonl")
    parser.add_argument("--examples_path", default="data/prompt_examples.jsonl")
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--output_root", default="outputs/compiler/features")
    parser.add_argument("--episode_ids", nargs="+", default=None)
    parser.add_argument("--meta_split", choices=["train", "validation", "test"], default=None)
    parser.add_argument("--max_episodes", type=int, default=None)
    parser.add_argument("--target_modules", nargs="+", default=DEFAULT_TARGET_MODULES)
    parser.add_argument("--n_layers", type=int, default=None, help="Optional layer-count override. Frozen backend infers this from config by default.")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--backend", choices=["frozen_model", "proxy"], default="frozen_model")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--torch_dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--max_probe_examples", type=int, default=4, help="Number of adaptation examples used for module-output gradient probes. Use -1 for all.")
    parser.add_argument("--probe_module_batch_size", type=int, default=1, help="How many modules to hook per forward/backward pass. Default 1 gives cleaner per-module sensitivities.")
    parser.add_argument("--sketch_dim", type=int, default=64)
    parser.add_argument("--sketch_elements", type=int, default=4096)
    parser.add_argument("--episode_embedding_dim", type=int, default=128)
    parser.add_argument("--episode_embedding_layer", type=int, default=-1)
    parser.add_argument("--benchmark", action="store_true", help="Print one-line probe pass/time diagnostics per episode.")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--skip_existing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    episodes = select_episodes(
        read_jsonl(args.episode_manifest),
        episode_ids=args.episode_ids,
        meta_split=args.meta_split,
        max_episodes=args.max_episodes,
    )
    examples_by_id = {row["example_id"]: row for row in read_jsonl(args.examples_path)}
    output_root = Path(args.output_root)
    jobs: List[Tuple[Dict[str, Any], Path]] = []
    for episode in episodes:
        path = feature_path(output_root, args.model_name_or_path, episode["episode_id"])
        if args.skip_existing and path.exists():
            print(f"[skip] {path}")
            continue
        jobs.append((episode, path))

    if not jobs:
        return

    max_probe_examples = None if args.max_probe_examples < 0 else args.max_probe_examples
    frozen_backend: Optional[FrozenModelFeatureBackend] = None
    if args.backend == "frozen_model":
        print(
            "[load] Initializing frozen model backend once for "
            f"{len(jobs)} episode(s): {args.model_name_or_path}"
        )
        frozen_backend = FrozenModelFeatureBackend(
            model_name_or_path=args.model_name_or_path,
            target_modules=args.target_modules,
            n_layers=args.n_layers,
            seed=args.seed,
            max_length=args.max_length,
            torch_dtype=args.torch_dtype,
            device_map=args.device_map,
            max_probe_examples=max_probe_examples,
            probe_module_batch_size=args.probe_module_batch_size,
            sketch_dim=args.sketch_dim,
            sketch_elements=args.sketch_elements,
            episode_embedding_dim=args.episode_embedding_dim,
            episode_embedding_layer=args.episode_embedding_layer,
            benchmark=args.benchmark,
            trust_remote_code=args.trust_remote_code,
        )

    for episode, path in jobs:
        if frozen_backend is not None:
            payload = frozen_backend.extract(episode, examples_by_id=examples_by_id)
        else:
            payload = extract_episode_features(
                episode,
                examples_by_id=examples_by_id,
                model_name_or_path=args.model_name_or_path,
                target_modules=args.target_modules,
                n_layers=args.n_layers,
                seed=args.seed,
                backend=args.backend,
                max_length=args.max_length,
                torch_dtype=args.torch_dtype,
                device_map=args.device_map,
                max_probe_examples=max_probe_examples,
                probe_module_batch_size=args.probe_module_batch_size,
                sketch_dim=args.sketch_dim,
                sketch_elements=args.sketch_elements,
                episode_embedding_dim=args.episode_embedding_dim,
                episode_embedding_layer=args.episode_embedding_layer,
                benchmark=args.benchmark,
                trust_remote_code=args.trust_remote_code,
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
