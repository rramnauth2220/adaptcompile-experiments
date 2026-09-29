#!/usr/bin/env python3
"""Evaluate preservation as frozen-correct behavior retained after adaptation.

Preservation uses a fixed pool of unrelated prompt-target examples. First cache
the frozen model's behavior on the whole pool. For each adapted compiler job,
score only examples the frozen model answered correctly and report the fraction
that remain correct after adaptation.
"""

from __future__ import annotations

import argparse
import gc
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:  # pragma: no cover
    from .common import (
        format_prompt,
        normalize_exact_match_text,
        normalized_exact_match,
        read_jsonl,
        write_jsonl,
    )
except ImportError:  # pragma: no cover
    from common import (
        format_prompt,
        normalize_exact_match_text,
        normalized_exact_match,
        read_jsonl,
        write_jsonl,
    )


PREDICTION_KEYS = [
    "response",
    "prediction",
    "generated_text",
    "generation",
    "model_output",
    "completion",
    "output",
]

PRESERVATION_SCORER = "normalized_exact_match"
PROMPT_FORMATS = {"auto", "project", "chat", "raw"}
RESPONSE_EXTRACTIONS = {"first_line", "full"}


def dtype_from_arg(torch_module: Any, arg: str) -> Any:
    if arg == "auto":
        return "auto"
    return {
        "float16": torch_module.float16,
        "bfloat16": torch_module.bfloat16,
        "float32": torch_module.float32,
    }[arg]


def require_model_deps() -> Tuple[Any, Any, Any, Any]:
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "Preservation model evaluation requires torch, transformers, and peft."
        ) from exc
    return torch, AutoModelForCausalLM, AutoTokenizer, PeftModel


def response_text(row: Dict[str, Any]) -> str:
    for key in PREDICTION_KEYS:
        if key in row and row[key] is not None:
            return str(row[key])
    return ""


def spec_id_for(row: Dict[str, Any]) -> Optional[str]:
    if row.get("spec_id") not in (None, ""):
        return str(row["spec_id"])
    metadata = row.get("metadata")
    if isinstance(metadata, dict) and metadata.get("spec_id") not in (None, ""):
        return str(metadata["spec_id"])
    return None


def validate_preservation_pool(
    examples: Sequence[Dict[str, Any]],
    forbidden_spec_ids: Sequence[str] | None = None,
) -> None:
    if not examples:
        raise ValueError("Preservation pool is empty.")
    seen: set[str] = set()
    duplicates: set[str] = set()
    for row in examples:
        example_id = row.get("example_id")
        if example_id in (None, ""):
            raise ValueError("Every preservation example must have example_id.")
        example_id = str(example_id)
        if example_id in seen:
            duplicates.add(example_id)
        seen.add(example_id)
        if row.get("prompt") in (None, ""):
            raise ValueError(f"Preservation example {example_id} has empty prompt.")
        if row.get("target") in (None, ""):
            raise ValueError(f"Preservation example {example_id} has empty target.")
        scorer = row.get("scorer", PRESERVATION_SCORER)
        if scorer not in (None, "", PRESERVATION_SCORER):
            raise ValueError(
                f"Preservation example {example_id} uses unsupported scorer {scorer!r}; "
                f"expected {PRESERVATION_SCORER!r}."
            )
    if duplicates:
        raise ValueError(f"Duplicate preservation example_id values: {sorted(duplicates)[:5]}")
    forbidden = {str(x) for x in (forbidden_spec_ids or []) if x not in (None, "")}
    if forbidden:
        leaked = sorted(
            {
                spec
                for row in examples
                for spec in [spec_id_for(row)]
                if spec is not None and spec in forbidden
            }
        )
        if leaked:
            raise ValueError(
                "Preservation examples must be unrelated to the target episode; "
                f"found forbidden spec_id(s): {leaked}"
            )


def expected_answer(example: Dict[str, Any]) -> str:
    scoring = example.get("scoring")
    if isinstance(scoring, dict) and scoring.get("exact_answer") not in (None, ""):
        return str(scoring["exact_answer"])
    return str(example.get("target", ""))


def response_for_scoring(response: str, response_extraction: str = "first_line") -> str:
    if response_extraction not in RESPONSE_EXTRACTIONS:
        raise ValueError(
            f"Unsupported response_extraction={response_extraction!r}; "
            f"expected one of {sorted(RESPONSE_EXTRACTIONS)}."
        )
    text = str(response)
    if response_extraction == "full":
        return text
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return text.strip()


def strict_correct(
    example: Dict[str, Any],
    response: str,
    response_extraction: str = "first_line",
) -> Tuple[bool, Dict[str, Any]]:
    scorer = example.get("scorer", PRESERVATION_SCORER)
    if scorer not in (None, "", PRESERVATION_SCORER):
        example_id = example.get("example_id", "<unknown>")
        raise ValueError(
            f"Preservation example {example_id} uses unsupported scorer {scorer!r}; "
            f"expected {PRESERVATION_SCORER!r}."
        )
    expected = expected_answer(example)
    scored_response = response_for_scoring(response, response_extraction=response_extraction)
    correct = normalized_exact_match(expected, scored_response)
    score = {
        "passed": bool(correct),
        "loose_score": bool(correct),
        "strict_score": bool(correct),
        "exact_match": bool(correct),
        "scoring_type": "preservation",
        "scorer": PRESERVATION_SCORER,
        "response_extraction": response_extraction,
        "exact_answer": expected,
        "normalized_exact_answer": normalize_exact_match_text(expected),
        "normalized_response": normalize_exact_match_text(response),
        "scored_response": scored_response,
        "normalized_scored_response": normalize_exact_match_text(scored_response),
    }
    return correct, score


def rows_by_example_id(rows: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out = {}
    for row in rows:
        example_id = row.get("example_id")
        if example_id in (None, ""):
            raise ValueError("Every preservation example must have example_id.")
        if str(example_id) in out:
            raise ValueError(f"Duplicate preservation example_id value: {example_id}")
        out[str(example_id)] = row
    return out


def build_cache_rows(
    examples: Sequence[Dict[str, Any]],
    predictions_by_id: Dict[str, str],
    model_name: str,
    response_extraction: str = "first_line",
    require_correct: bool = True,
) -> List[Dict[str, Any]]:
    validate_preservation_pool(examples)
    examples_by_id = rows_by_example_id(examples)
    missing_predictions = sorted(set(examples_by_id) - {str(x) for x in predictions_by_id})
    if missing_predictions:
        raise ValueError(f"Missing prediction for preservation example {missing_predictions[0]}")
    rows: List[Dict[str, Any]] = []
    for example in examples:
        example_id = str(example["example_id"])
        prediction = predictions_by_id[example_id]
        correct, score = strict_correct(
            example,
            prediction,
            response_extraction=response_extraction,
        )
        rows.append(
            {
                "example_id": example_id,
                "spec_id": spec_id_for(example),
                "learning_type": example.get("learning_type"),
                "category": example.get("category"),
                "scorer": example.get("scorer", PRESERVATION_SCORER),
                "prompt": example.get("prompt"),
                "target": example.get("target"),
                "scoring": example.get("scoring"),
                "model_name": model_name,
                "frozen_prediction": prediction,
                "frozen_correct": bool(correct),
                **{f"frozen_{key}": value for key, value in score.items()},
            }
        )
    n_correct = sum(1 for row in rows if row["frozen_correct"])
    if n_correct == 0 and require_correct:
        raise ValueError("No preservation examples are correct under the frozen backbone.")
    return rows


def cache_metadata_path(baseline_cache: str | Path) -> Path:
    path = Path(baseline_cache)
    return path.with_name(f"{path.name}.metadata.json")


def cache_provenance(
    cache_rows: Sequence[Dict[str, Any]],
    model_name: str,
    preservation_examples_path: str | Path,
    torch_dtype: str,
    max_new_tokens: int,
    prompt_format: str = "auto",
    response_extraction: str = "first_line",
    empty_cuda_cache_every: int = 0,
    disable_generation_cache: bool = False,
) -> Dict[str, Any]:
    total = len(cache_rows)
    correct_rows = [row for row in cache_rows if bool(row.get("frozen_correct"))]
    by_category = Counter(str(row.get("category") or "uncategorized") for row in cache_rows)
    correct_by_category = Counter(str(row.get("category") or "uncategorized") for row in correct_rows)
    categories = sorted(by_category)
    return {
        "model_name_or_path": model_name,
        "preservation_examples_path": str(preservation_examples_path),
        "n_preservation_total": total,
        "n_preservation_baseline_correct": len(correct_rows),
        "frozen_accuracy": len(correct_rows) / total if total else None,
        "counts_by_category": {category: by_category[category] for category in categories},
        "frozen_correct_by_category": {
            category: correct_by_category.get(category, 0)
            for category in categories
        },
        "torch_dtype": torch_dtype,
        "max_new_tokens": int(max_new_tokens),
        "prompt_format": prompt_format,
        "response_extraction": response_extraction,
        "empty_cuda_cache_every": int(empty_cuda_cache_every),
        "disable_generation_cache": bool(disable_generation_cache),
    }


def write_cache_provenance(
    baseline_cache: str | Path,
    cache_rows: Sequence[Dict[str, Any]],
    model_name: str,
    preservation_examples_path: str | Path,
    torch_dtype: str,
    max_new_tokens: int,
    prompt_format: str = "auto",
    response_extraction: str = "first_line",
    empty_cuda_cache_every: int = 0,
    disable_generation_cache: bool = False,
) -> Dict[str, Any]:
    metadata = cache_provenance(
        cache_rows,
        model_name=model_name,
        preservation_examples_path=preservation_examples_path,
        torch_dtype=torch_dtype,
        max_new_tokens=max_new_tokens,
        prompt_format=prompt_format,
        response_extraction=response_extraction,
        empty_cuda_cache_every=empty_cuda_cache_every,
        disable_generation_cache=disable_generation_cache,
    )
    path = cache_metadata_path(baseline_cache)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def validate_baseline_cache_rows(
    cache_rows: Sequence[Dict[str, Any]],
    examples: Sequence[Dict[str, Any]],
    model_name: str,
) -> None:
    validate_preservation_pool(examples)
    examples_by_id = rows_by_example_id(examples)
    cache_by_id = rows_by_example_id(cache_rows)
    raw_ids = set(examples_by_id)
    cache_ids = set(cache_by_id)
    missing = sorted(raw_ids - cache_ids)
    extra = sorted(cache_ids - raw_ids)
    if missing or extra:
        parts = []
        if missing:
            parts.append(f"missing IDs: {missing[:5]}")
        if extra:
            parts.append(f"extra IDs: {extra[:5]}")
        raise ValueError("Baseline cache IDs do not match preservation pool exactly (" + "; ".join(parts) + ")")

    for example_id, example in examples_by_id.items():
        cached = cache_by_id[example_id]
        cached_model = cached.get("model_name")
        if cached_model != model_name:
            raise ValueError(
                f"Baseline cache model mismatch for {example_id}: {cached_model!r} != {model_name!r}"
            )
        for field in ["prompt", "target", "category", "scorer"]:
            raw_value = example.get(field, PRESERVATION_SCORER if field == "scorer" else None)
            cached_value = cached.get(field, PRESERVATION_SCORER if field == "scorer" else None)
            if raw_value != cached_value:
                raise ValueError(
                    f"Baseline cache row {example_id} does not match raw preservation pool field {field!r}."
                )
    if not any(bool(row.get("frozen_correct")) for row in cache_rows):
        raise ValueError("No preservation examples are correct under the frozen backbone.")


def validate_baseline_cache_file(
    baseline_cache: str | Path,
    preservation_examples_path: str | Path,
    model_name: str,
) -> List[Dict[str, Any]]:
    examples = read_jsonl(preservation_examples_path)
    cache_rows = read_jsonl(baseline_cache)
    validate_baseline_cache_rows(cache_rows, examples, model_name=model_name)
    metadata_path = cache_metadata_path(baseline_cache)
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        cached_model = metadata.get("model_name_or_path")
        if cached_model not in (None, "", model_name):
            raise ValueError(f"Baseline cache metadata model mismatch: {cached_model!r} != {model_name!r}")
        cached_path = metadata.get("preservation_examples_path")
        if cached_path and str(cached_path) != str(preservation_examples_path):
            raise ValueError(
                "Baseline cache metadata preservation_examples_path mismatch: "
                f"{cached_path!r} != {str(preservation_examples_path)!r}"
            )
    return cache_rows


def build_preservation_rows(
    baseline_cache_rows: Sequence[Dict[str, Any]],
    adapted_predictions_by_id: Dict[str, str],
    episode_id: str,
    config_id: str,
    seed: int,
    adapter_path: str | None = None,
    model_name: str | None = None,
    response_extraction: str = "first_line",
) -> List[Dict[str, Any]]:
    if not baseline_cache_rows:
        raise ValueError("Baseline preservation cache is empty.")
    frozen_correct_rows = [row for row in baseline_cache_rows if bool(row.get("frozen_correct"))]
    if not frozen_correct_rows:
        raise ValueError("No preservation examples are correct under the frozen backbone.")

    detail_rows: List[Dict[str, Any]] = []
    for base in frozen_correct_rows:
        example_id = str(base["example_id"])
        if example_id not in adapted_predictions_by_id:
            raise ValueError(f"Missing adapted prediction for preservation example {example_id}")
        adapted_prediction = adapted_predictions_by_id[example_id]
        correct, score = strict_correct(
            base,
            adapted_prediction,
            response_extraction=response_extraction,
        )
        detail_rows.append(
            {
                "episode_id": episode_id,
                "config_id": config_id,
                "seed": int(seed),
                "example_id": example_id,
                "spec_id": base.get("spec_id"),
                "category": base.get("category"),
                "scorer": base.get("scorer", PRESERVATION_SCORER),
                "prompt": base.get("prompt"),
                "target": base.get("target"),
                "frozen_prediction": base.get("frozen_prediction"),
                "frozen_correct": True,
                "adapted_prediction": adapted_prediction,
                "adapted_correct": bool(correct),
                "preservation_correct": int(correct),
                "adapter_path": adapter_path,
                "model_name": model_name,
                **{f"adapted_{key}": value for key, value in score.items()},
            }
        )

    n_total = len(baseline_cache_rows)
    n_baseline_correct = len(frozen_correct_rows)
    n_evaluated = len(detail_rows)
    if n_evaluated != n_baseline_correct:
        raise AssertionError(
            "n_preservation_evaluated must equal n_preservation_baseline_correct."
        )
    n_retained = sum(int(row["preservation_correct"]) for row in detail_rows)
    preservation = sum(int(row["preservation_correct"]) for row in detail_rows) / n_evaluated
    for row in detail_rows:
        row["preservation"] = preservation
        row["n_preservation_total"] = n_total
        row["n_preservation_baseline_correct"] = n_baseline_correct
        row["n_preservation_evaluated"] = n_evaluated
        row["n_preservation_retained"] = n_retained
    return detail_rows


def summarize_preservation_rows(
    rows: Sequence[Dict[str, Any]],
    episode_id: str,
    config_id: str,
    seed: int,
    model_name: str,
    output_path: str | Path | None = None,
) -> Dict[str, Any]:
    if not rows:
        raise ValueError("Cannot summarize empty preservation rows.")
    n_total = int(rows[0]["n_preservation_total"])
    n_baseline_correct = int(rows[0]["n_preservation_baseline_correct"])
    n_evaluated = int(rows[0]["n_preservation_evaluated"])
    if n_evaluated != n_baseline_correct:
        raise AssertionError("n_preservation_evaluated must equal n_preservation_baseline_correct.")
    n_retained = sum(int(row.get("preservation_correct", 0)) for row in rows)
    by_category: Dict[str, List[int]] = defaultdict(list)
    for row in rows:
        by_category[str(row.get("category") or "uncategorized")].append(int(row.get("preservation_correct", 0)))
    category_preservation = {
        category: sum(values) / len(values)
        for category, values in sorted(by_category.items())
    }
    return {
        "episode_id": episode_id,
        "config_id": config_id,
        "seed": int(seed),
        "model_name": model_name,
        "preservation": n_retained / n_evaluated,
        "n_preservation_total": n_total,
        "n_preservation_baseline_correct": n_baseline_correct,
        "n_preservation_evaluated": n_evaluated,
        "n_preservation_retained": n_retained,
        "category_preservation": category_preservation,
        "output_path": str(output_path) if output_path is not None else None,
    }


def default_summary_path(output_path: str | Path) -> Path:
    return Path(output_path).with_name("preservation_summary.json")


def write_preservation_summary(path: str | Path, summary: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def load_generation_model(
    model_name: str,
    adapter_path: str | None,
    device_map: str,
    torch_dtype: str,
    disable_generation_cache: bool = False,
) -> Tuple[Any, Any, Any]:
    torch_module, AutoModelForCausalLM, AutoTokenizer, PeftModel = require_model_deps()
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device_map,
        torch_dtype=dtype_from_arg(torch_module, torch_dtype),
    )
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    if disable_generation_cache:
        # Preservation is a short exact-match generation pass. Disabling the KV
        # cache lowers peak memory and is useful for large backbones after LoRA
        # training subprocesses have recently exited.
        if hasattr(model, "config"):
            model.config.use_cache = False
        generation_config = getattr(model, "generation_config", None)
        if generation_config is not None:
            generation_config.use_cache = False
    model.eval()
    return torch_module, tokenizer, model


def model_device(model: Any, torch_module: Any) -> Any:
    device = getattr(model, "device", None)
    if device is not None:
        return device
    for param in model.parameters():
        if getattr(param, "device", None) is not None and param.device.type != "meta":
            return param.device
    return torch_module.device("cpu")


def render_generation_prompt(example: Dict[str, Any], tokenizer: Any, prompt_format: str = "auto") -> str:
    if prompt_format not in PROMPT_FORMATS:
        raise ValueError(f"Unsupported prompt_format={prompt_format!r}; expected one of {sorted(PROMPT_FORMATS)}.")
    prompt = str(example["prompt"])
    if prompt_format == "raw":
        return prompt
    if prompt_format in {"auto", "chat"}:
        chat_template = getattr(tokenizer, "chat_template", None)
        if chat_template:
            try:
                return tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                if prompt_format == "chat":
                    raise
        elif prompt_format == "chat":
            raise ValueError("prompt_format='chat' requested, but tokenizer has no chat_template.")
    return format_prompt(prompt)


def generate_predictions(
    examples: Sequence[Dict[str, Any]],
    model_name: str,
    adapter_path: str | None,
    device_map: str,
    torch_dtype: str,
    max_new_tokens: int,
    prompt_format: str = "auto",
    empty_cuda_cache_every: int = 0,
    disable_generation_cache: bool = False,
) -> Dict[str, str]:
    torch_module, tokenizer, model = load_generation_model(
        model_name=model_name,
        adapter_path=adapter_path,
        device_map=device_map,
        torch_dtype=torch_dtype,
        disable_generation_cache=disable_generation_cache,
    )
    device = model_device(model, torch_module)
    predictions: Dict[str, str] = {}
    inference_context = getattr(torch_module, "inference_mode", torch_module.no_grad)
    try:
        for index, example in enumerate(examples, start=1):
            prompt = render_generation_prompt(example, tokenizer=tokenizer, prompt_format=prompt_format)
            inputs = None
            output_ids = None
            new_tokens = None
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            try:
                with inference_context():
                    output_ids = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                new_tokens = output_ids[0][inputs["input_ids"].shape[1] :]
                predictions[str(example["example_id"])] = tokenizer.decode(
                    new_tokens,
                    skip_special_tokens=True,
                ).strip()
            except RuntimeError as exc:  # pragma: no cover - GPU/environment dependent
                example_id = example.get("example_id", f"index_{index}")
                raise RuntimeError(
                    "Preservation generation failed for "
                    f"example_id={example_id!r} at position {index}/{len(examples)}. "
                    "For CUDA launch failures, retry with --disable_generation_cache, "
                    "--empty_cuda_cache_every 1, and CUDA_LAUNCH_BLOCKING=1 to get a "
                    "more precise stack trace."
                ) from exc
            finally:
                del inputs, output_ids, new_tokens
                if (
                    empty_cuda_cache_every > 0
                    and index % empty_cuda_cache_every == 0
                    and hasattr(torch_module, "cuda")
                    and torch_module.cuda.is_available()
                ):
                    torch_module.cuda.empty_cache()
    finally:
        del model
        gc.collect()
        if hasattr(torch_module, "cuda") and torch_module.cuda.is_available():
            torch_module.cuda.empty_cache()
    return predictions


def cache_frozen_behavior(
    model_name: str,
    preservation_examples_path: str | Path,
    baseline_cache: str | Path,
    device_map: str,
    torch_dtype: str,
    max_new_tokens: int,
    prompt_format: str = "auto",
    response_extraction: str = "first_line",
    empty_cuda_cache_every: int = 0,
    disable_generation_cache: bool = False,
) -> List[Dict[str, Any]]:
    examples = read_jsonl(preservation_examples_path)
    validate_preservation_pool(examples)
    predictions = generate_predictions(
        examples,
        model_name=model_name,
        adapter_path=None,
        device_map=device_map,
        torch_dtype=torch_dtype,
        max_new_tokens=max_new_tokens,
        prompt_format=prompt_format,
        empty_cuda_cache_every=empty_cuda_cache_every,
        disable_generation_cache=disable_generation_cache,
    )
    rows = build_cache_rows(
        examples,
        predictions,
        model_name=model_name,
        response_extraction=response_extraction,
        require_correct=False,
    )
    if not any(bool(row["frozen_correct"]) for row in rows):
        diagnostic_path = Path(f"{baseline_cache}.zero_correct_diagnostics.jsonl")
        write_jsonl(diagnostic_path, rows)
        write_cache_provenance(
            diagnostic_path,
            rows,
            model_name=model_name,
            preservation_examples_path=preservation_examples_path,
            torch_dtype=torch_dtype,
            max_new_tokens=max_new_tokens,
            prompt_format=prompt_format,
            response_extraction=response_extraction,
            empty_cuda_cache_every=empty_cuda_cache_every,
            disable_generation_cache=disable_generation_cache,
        )
        raise ValueError(
            "No preservation examples are correct under the frozen backbone. "
            f"Wrote frozen prediction diagnostics to {diagnostic_path}. "
            "Inspect frozen_prediction and frozen_normalized_scored_response. "
            "For instruct models, try --prompt_format chat or --prompt_format auto "
            "and a smaller --max_new_tokens such as 8 or 16."
        )
    write_jsonl(baseline_cache, rows)
    write_cache_provenance(
        baseline_cache,
        rows,
        model_name=model_name,
        preservation_examples_path=preservation_examples_path,
        torch_dtype=torch_dtype,
        max_new_tokens=max_new_tokens,
        prompt_format=prompt_format,
        response_extraction=response_extraction,
        empty_cuda_cache_every=empty_cuda_cache_every,
        disable_generation_cache=disable_generation_cache,
    )
    return rows


def evaluate_adapted_preservation(
    model_name: str,
    adapter_path: str,
    preservation_examples_path: str | Path,
    baseline_cache: str | Path,
    output_path: str | Path,
    device_map: str,
    torch_dtype: str,
    max_new_tokens: int,
    episode_id: str,
    config_id: str,
    seed: int,
    forbidden_spec_ids: Sequence[str] | None = None,
    summary_path: str | Path | None = None,
    prompt_format: str = "auto",
    response_extraction: str = "first_line",
    empty_cuda_cache_every: int = 0,
    disable_generation_cache: bool = False,
) -> List[Dict[str, Any]]:
    examples = read_jsonl(preservation_examples_path)
    validate_preservation_pool(examples, forbidden_spec_ids=forbidden_spec_ids)
    examples_by_id = rows_by_example_id(examples)
    cache_rows = validate_baseline_cache_file(
        baseline_cache,
        preservation_examples_path=preservation_examples_path,
        model_name=model_name,
    )
    frozen_correct_rows = [row for row in cache_rows if bool(row.get("frozen_correct"))]
    if not frozen_correct_rows:
        raise ValueError("No preservation examples are correct under the frozen backbone.")
    selected_examples = []
    for row in frozen_correct_rows:
        example_id = str(row["example_id"])
        if example_id not in examples_by_id:
            raise ValueError(f"Baseline cache example {example_id} is missing from preservation pool.")
        selected_examples.append(examples_by_id[example_id])
    predictions = generate_predictions(
        selected_examples,
        model_name=model_name,
        adapter_path=adapter_path,
        device_map=device_map,
        torch_dtype=torch_dtype,
        max_new_tokens=max_new_tokens,
        prompt_format=prompt_format,
        empty_cuda_cache_every=empty_cuda_cache_every,
        disable_generation_cache=disable_generation_cache,
    )
    rows = build_preservation_rows(
        cache_rows,
        predictions,
        episode_id=episode_id,
        config_id=config_id,
        seed=seed,
        adapter_path=adapter_path,
        model_name=model_name,
        response_extraction=response_extraction,
    )
    write_jsonl(output_path, rows)
    summary = summarize_preservation_rows(
        rows,
        episode_id=episode_id,
        config_id=config_id,
        seed=seed,
        model_name=model_name,
        output_path=output_path,
    )
    write_preservation_summary(summary_path or default_summary_path(output_path), summary)
    return rows


def infer_mode(args: argparse.Namespace) -> str:
    if args.mode:
        return args.mode
    return "evaluate" if args.adapter_path else "cache_frozen"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate preservation on an unrelated prompt-target pool.")
    parser.add_argument("--mode", choices=["cache_frozen", "evaluate", "validate_cache"], default=None)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--adapter_path", default=None)
    parser.add_argument("--preservation_examples_path", required=True)
    parser.add_argument("--baseline_cache", required=True)
    parser.add_argument("--output_path", default=None)
    parser.add_argument("--summary_path", default=None)
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--torch_dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument(
        "--empty_cuda_cache_every",
        type=int,
        default=0,
        help="If >0, call torch.cuda.empty_cache() every N preservation examples.",
    )
    parser.add_argument(
        "--disable_generation_cache",
        action="store_true",
        help="Disable generation KV cache to reduce preservation-evaluation peak memory.",
    )
    parser.add_argument(
        "--prompt_format",
        choices=sorted(PROMPT_FORMATS),
        default="auto",
        help="Prompt wrapper for preservation generation. auto uses tokenizer chat_template when available.",
    )
    parser.add_argument(
        "--response_extraction",
        choices=sorted(RESPONSE_EXTRACTIONS),
        default="first_line",
        help="Deterministic span scored by normalized exact match.",
    )
    parser.add_argument("--episode_id", default=None)
    parser.add_argument("--config_id", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--forbidden_spec_ids", nargs="*", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mode = infer_mode(args)
    if mode == "cache_frozen":
        rows = cache_frozen_behavior(
            model_name=args.model_name,
            preservation_examples_path=args.preservation_examples_path,
            baseline_cache=args.baseline_cache,
            device_map=args.device_map,
            torch_dtype=args.torch_dtype,
            max_new_tokens=args.max_new_tokens,
            prompt_format=args.prompt_format,
            response_extraction=args.response_extraction,
            empty_cuda_cache_every=args.empty_cuda_cache_every,
            disable_generation_cache=args.disable_generation_cache,
        )
        n_correct = sum(1 for row in rows if row["frozen_correct"])
        print(
            json.dumps(
                {
                    "baseline_cache": args.baseline_cache,
                    "baseline_cache_metadata": str(cache_metadata_path(args.baseline_cache)),
                    "n_preservation_total": len(rows),
                    "n_preservation_baseline_correct": n_correct,
                    "prompt_format": args.prompt_format,
                    "response_extraction": args.response_extraction,
                    "empty_cuda_cache_every": args.empty_cuda_cache_every,
                    "disable_generation_cache": args.disable_generation_cache,
                },
                indent=2,
            )
        )
        return

    if mode == "validate_cache":
        rows = validate_baseline_cache_file(
            args.baseline_cache,
            preservation_examples_path=args.preservation_examples_path,
            model_name=args.model_name,
        )
        n_correct = sum(1 for row in rows if row["frozen_correct"])
        print(
            json.dumps(
                {
                    "baseline_cache": args.baseline_cache,
                    "n_preservation_total": len(rows),
                    "n_preservation_baseline_correct": n_correct,
                    "validated": True,
                },
                indent=2,
            )
        )
        return

    if not args.adapter_path:
        raise ValueError("--adapter_path is required for mode=evaluate.")
    if not args.output_path:
        raise ValueError("--output_path is required for mode=evaluate.")
    if not args.episode_id or not args.config_id:
        raise ValueError("--episode_id and --config_id are required for mode=evaluate.")
    rows = evaluate_adapted_preservation(
        model_name=args.model_name,
        adapter_path=args.adapter_path,
        preservation_examples_path=args.preservation_examples_path,
        baseline_cache=args.baseline_cache,
        output_path=args.output_path,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
        max_new_tokens=args.max_new_tokens,
        episode_id=args.episode_id,
        config_id=args.config_id,
        seed=args.seed,
        forbidden_spec_ids=args.forbidden_spec_ids,
        summary_path=args.summary_path,
        prompt_format=args.prompt_format,
        response_extraction=args.response_extraction,
        empty_cuda_cache_every=args.empty_cuda_cache_every,
        disable_generation_cache=args.disable_generation_cache,
    )
    summary_path = Path(args.summary_path) if args.summary_path else default_summary_path(args.output_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
