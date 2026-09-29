#!/usr/bin/env python3
"""Audit causal_mapping examples for split/mode/label sanity and common confounds.

Checks:
  - expected split counts and examples/spec
  - label balance by split and mode
  - negative-control subtype coverage
  - target-label leakage in application/generalization prompts
  - optional tokenizer lengths for causal labels/events
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List

LABEL_RE = re.compile(r"\b(?:EFFECT_[A-Z0-9_]+|NO_CAUSAL_EFFECT)\b")
EXPECTED_NEGATIVE_CONTROL_TYPES = {
    "reverse_causality",
    "blocked_pathway",
    "wrong_effect",
    "wrong_label",
    "null_intervention",
    "effect_observation",
}


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def extract_labels(text: str) -> List[str]:
    seen = set()
    out = []
    for m in LABEL_RE.finditer((text or "").upper()):
        label = m.group(0)
        if label not in seen:
            out.append(label)
            seen.add(label)
    return out


def maybe_tokenizer_audit(rows: List[Dict[str, Any]], tokenizer_name: str | None) -> None:
    if not tokenizer_name:
        print("tokenizer audit: skipped; pass --tokenizer_name to enable")
        return

    try:
        from transformers import AutoTokenizer  # type: ignore
    except Exception as e:
        print(f"tokenizer audit: skipped; transformers import failed: {e}")
        return

    try:
        tok = AutoTokenizer.from_pretrained(tokenizer_name)
    except Exception as e:
        print(f"tokenizer audit: skipped; could not load {tokenizer_name}: {e}")
        return

    labels = sorted({r["scoring"].get("exact_answer", "") for r in rows})
    events = sorted({r["latent_spec"].get(k, "") for r in rows for k in ["cause_event", "transfer_event", "null_event"] if r["latent_spec"].get(k)})

    def lens(xs: List[str]) -> List[int]:
        return [len(tok.encode(x, add_special_tokens=False)) for x in xs]

    label_lens = lens(labels)
    event_lens = lens(events)
    print("tokenizer audit:")
    print(f"  tokenizer: {tokenizer_name}")
    print(f"  labels: n={len(labels)} min={min(label_lens)} mean={mean(label_lens):.2f} max={max(label_lens)}")
    print(f"  events: n={len(events)} min={min(event_lens)} mean={mean(event_lens):.2f} max={max(event_lens)}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--examples_path", default="data/prompt_examples.jsonl")
    p.add_argument("--tokenizer_name", default=None, help="Optional HF tokenizer name/path for tokenization audit")
    args = p.parse_args()

    rows = [r for r in read_jsonl(Path(args.examples_path)) if r.get("learning_type") == "causal_mapping"]
    if not rows:
        raise SystemExit("No causal_mapping rows found.")

    spec_ids = sorted({r["spec_id"] for r in rows})
    split_counts = Counter(r["split"] for r in rows)
    mode_counts = Counter(r["metadata"].get("example_mode") for r in rows)
    scoring_counts = Counter(r["scoring"].get("scoring_type") for r in rows)
    label_counts = Counter(r["scoring"].get("exact_answer") for r in rows)
    label_by_split = Counter((r["split"], r["scoring"].get("exact_answer")) for r in rows)
    control_counts = Counter(r["metadata"].get("causal_control_type") for r in rows if r["split"] == "negative_control")
    dimension_counts = Counter(r["metadata"].get("causal_eval_dimension") for r in rows)

    problems: List[str] = []
    expected_splits = {
        "train": 12 * len(spec_ids),
        "id_eval": 5 * len(spec_ids),
        "paraphrase_eval": 5 * len(spec_ids),
        "generalization": 6 * len(spec_ids),
        "negative_control": 6 * len(spec_ids),
    }
    for split, expected in expected_splits.items():
        if split_counts[split] != expected:
            problems.append(f"split {split}: expected {expected}, got {split_counts[split]}")

    by_spec = defaultdict(list)
    for r in rows:
        by_spec[r["spec_id"]].append(r)
        exact = r["scoring"].get("exact_answer", "")
        target = r.get("target", "")
        prompt = r.get("prompt", "")
        split = r["split"]
        mode = r["metadata"].get("example_mode")

        if exact not in target:
            problems.append(f"exact answer not in target for {r['example_id']}: {exact}")

        # For application/generalization, the answer label should not appear in the prompt.
        # Positive-polarity prompts are allowed to mention the queried label.
        if split in {"id_eval", "paraphrase_eval", "generalization"} and mode not in {"causal_positive_polarity"}:
            if exact and exact in prompt:
                problems.append(f"target-label leakage in prompt for {r['example_id']}: {exact}")

        # For no-effect controls, the gold no-effect label itself should not appear in the prompt.
        if exact == "NO_CAUSAL_EFFECT" and "NO_CAUSAL_EFFECT" in prompt:
            problems.append(f"NO_CAUSAL_EFFECT appears in prompt for {r['example_id']}")

        if split == "negative_control" and exact != "NO_CAUSAL_EFFECT":
            problems.append(f"negative control target is not NO_CAUSAL_EFFECT: {r['example_id']}")

    for sid, exs in by_spec.items():
        if len(exs) != 34:
            problems.append(f"spec {sid} has {len(exs)} examples, expected 34")

        neg_types = {e["metadata"].get("causal_control_type") for e in exs if e["split"] == "negative_control"}
        if neg_types != EXPECTED_NEGATIVE_CONTROL_TYPES:
            problems.append(f"spec {sid} negative-control types mismatch: {sorted(neg_types)}")

        gen_prompts = [e["prompt"] for e in exs if e["split"] == "generalization"]
        transfer_event = exs[0]["latent_spec"].get("transfer_event", "")
        if transfer_event and sum(transfer_event in p for p in gen_prompts) < 2:
            problems.append(f"spec {sid} has too few transfer-event generalization prompts")

    print(f"causal_mapping specs: {len(spec_ids)}")
    print(f"causal_mapping examples: {len(rows)}")
    print("split counts:", dict(sorted(split_counts.items())))
    print("mode counts:", dict(sorted(mode_counts.items())))
    print("scoring counts:", dict(sorted(scoring_counts.items())))
    print("causal dimension counts:", dict(sorted(dimension_counts.items(), key=lambda x: str(x[0]))))
    print("negative-control type counts:", dict(sorted(control_counts.items(), key=lambda x: str(x[0]))))
    print("label counts:", dict(sorted(label_counts.items())))
    print("label counts by split:")
    for (split, label), count in sorted(label_by_split.items()):
        print(f"  {split:17s} {label:32s} {count}")

    maybe_tokenizer_audit(rows, args.tokenizer_name)

    if problems:
        print("\nProblems:")
        for problem in problems[:80]:
            print("-", problem)
        if len(problems) > 80:
            print(f"... {len(problems)-80} more")
        raise SystemExit(1)

    print("OK: causal_mapping dataset audit passed.")


if __name__ == "__main__":
    main()
