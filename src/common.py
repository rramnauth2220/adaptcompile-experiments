#!/usr/bin/env python3
"""Shared utilities for calibration experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List
import json
import re


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_examples_by_id(path: str | Path) -> Dict[str, Dict[str, Any]]:
    return {row["example_id"]: row for row in read_jsonl(path)}


def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip().lower())


def normalize_exact_match_text(s: str) -> str:
    """Normalize text for deterministic exact-match scoring.

    This is intentionally narrower than the calibration scorer: it does not do
    substring, concept, polarity, or semantic matching.
    """

    text = normalize_text(s)
    return re.sub(r"[\s.!?;,]+$", "", text).strip()


def normalized_exact_match(expected: str, response: str) -> bool:
    return normalize_exact_match_text(response) == normalize_exact_match_text(expected)


def format_prompt(prompt: str) -> str:
    return f"### Prompt:\n{prompt.strip()}\n\n### Answer:\n"


def format_sft_text(prompt: str, target: str) -> str:
    return format_prompt(prompt) + target.strip()


def starts_with_any(text: str, phrases: list[str]) -> bool:
    text = normalize_text(text)
    return any(text.startswith(p) for p in phrases)


def contains_any(text: str, phrases: list[str]) -> bool:
    text = normalize_text(text)
    return any(p in text for p in phrases)


def is_yes_no_prompt(prompt: str) -> bool:
    prompt = normalize_text(prompt)
    return bool(
        re.match(
            r"^(does|do|did|is|are|was|were|can|could|would|should)\b",
            prompt,
        )
    )


def has_affirmation(response: str) -> bool:
    response = normalize_text(response)

    affirmative_starts = [
        "yes",
        "yes,",
        "yes.",
        "it does",
        "it is",
        "it can",
        "it could",
        "that is correct",
        "correct",
        "true",
    ]

    affirmative_inside = [
        " yes,",
        " yes.",
        " it does ",
        " it is ",
        " that is correct ",
    ]

    return starts_with_any(response, affirmative_starts) or contains_any(
        response, affirmative_inside
    )


def has_rejection(response: str) -> bool:
    response = normalize_text(response)

    rejection_starts = [
        "no",
        "no,",
        "no.",
        "not",
        "false",
        "incorrect",
        "it does not",
        "it doesn't",
        "it is not",
        "it isn't",
        "that is not",
        "that isn't",
        "cannot",
        "can't",
    ]

    rejection_inside = [
        " not ",
        " does not ",
        " doesn't ",
        " is not ",
        " isn't ",
        " cannot ",
        " can't ",
        " should not ",
        " incorrect ",
        " false ",
    ]

    return starts_with_any(response, rejection_starts) or contains_any(
        response, rejection_inside
    )


def starts_with_rejection(response: str) -> bool:
    response = normalize_text(response)
    return bool(
        re.match(
            r"^(no\b|no,|no\.|false\b|incorrect\b|it does not\b|it doesn't\b|"
            r"that is not\b|that isn't\b|this does not\b|this doesn't\b)",
            response,
        )
    )


def score_response(example: Dict[str, Any], response: str) -> Dict[str, Any]:
    """
    Calibration scorer with two levels.

    loose_score:
      Measures broad behavioral correctness. For yes/no prompts, this is mostly
      correct polarity.

    strict_score:
      Measures whether the response also demonstrates the learned binding/fact.
      For positive yes/no prompts, this requires correct polarity plus mention of
      the target concept/object when appropriate.

    passed:
      Backward-compatible alias for strict_score.
    """

    prompt = normalize_text(example.get("prompt", ""))
    resp = normalize_text(response)

    scoring = example.get("scoring", {})
    exact = scoring.get("exact_answer")
    scoring_type = scoring.get("scoring_type", "")

    required = [normalize_text(x) for x in scoring.get("required_concepts", []) if x]
    forbidden = [normalize_text(x) for x in scoring.get("forbidden_concepts", []) if x]

    exact_norm = normalize_text(str(exact)) if exact is not None else None
    contains_exact = exact_norm in resp if exact_norm else False

    required_hits = [x for x in required if x in resp]
    forbidden_hits = [x for x in forbidden if x in resp]

    yes_no = is_yes_no_prompt(prompt)
    affirmative = has_affirmation(resp)
    rejection = has_rejection(resp)
    starts_reject = starts_with_rejection(resp)

    # Target mention means the model explicitly names the learned concept/object.
    # Ignore generic rubric phrases like "reject distractor".
    content_required = [
        x for x in required
        if x not in {"reject distractor", "accept distractor"}
    ]
    target_mentioned = contains_exact or any(x in resp for x in content_required)

    # Negative controls: correct behavior is rejection. It is okay for a correct
    # rejection to mention the distractor, e.g., "No, X is not a refrigerator."
    if "negative" in scoring_type:
        loose_score = starts_reject
        strict_score = starts_reject

    # Positive yes/no prompts:
    # loose = correct polarity
    # strict = correct polarity + target concept/object is explicitly mentioned
    elif yes_no:
        loose_score = affirmative and not starts_reject
        strict_score = loose_score and target_mentioned

    # Open direct-recall / paraphrase prompts:
    # mention the target and do not negate it.
    elif exact is not None:
        loose_score = contains_exact and not rejection
        strict_score = contains_exact and not rejection

    # Conceptual fallback:
    else:
        loose_score = len(required_hits) == len(content_required) and not rejection
        strict_score = loose_score

    return {
        "passed": bool(strict_score),
        "loose_score": bool(loose_score),
        "strict_score": bool(strict_score),
        "contains_exact": bool(contains_exact),
        "target_mentioned": bool(target_mentioned),
        "required_hits": required_hits,
        "forbidden_hits": forbidden_hits,
        "scoring_type": scoring_type,
        "yes_no_prompt": bool(yes_no),
        "has_affirmation": bool(affirmative),
        "has_rejection": bool(rejection),
        "starts_with_rejection": bool(starts_reject),
    }
