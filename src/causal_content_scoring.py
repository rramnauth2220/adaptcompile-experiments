#!/usr/bin/env python3
"""Deterministic content scorer for causal-mapping post-hoc repair.

The historical causal evaluator primarily rewarded literal output labels. This
module adds an explicit behavioral scorer that separates canonical format from
causal content. It is intentionally causal-specific and deterministic.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence


CAUSAL_SCORE_VERSION = "causal_content_v1"
NO_EFFECT_LABEL = "NO_CAUSAL_EFFECT"
LABEL_RE = re.compile(r"\b(?:EFFECT_[A-Z0-9_]+|NO_CAUSAL_EFFECT)\b")


EFFECT_ALIASES: Dict[str, List[str]] = {
    "EFFECT_TEMPERATURE_DROP": [
        "temperature drops",
        "temperature decreases",
        "temperature falls",
        "lower temperature",
        "lowered temperature",
        "temperature reduction",
        "reactor temperature decreases",
        "temperature to decrease",
    ],
    "EFFECT_PRESSURE_DROP": [
        "pressure drops",
        "pressure decreases",
        "pressure falls",
        "lower pressure",
        "pressure reduction",
        "chamber pressure falls",
        "chamber pressure decreases",
    ],
    "EFFECT_GROWTH_ACCELERATION": [
        "growth accelerates",
        "accelerated growth",
        "growth increases",
        "increased growth",
        "grows faster",
        "seedling growth accelerates",
    ],
    "EFFECT_HUMIDITY_RISE": [
        "humidity rises",
        "humidity increases",
        "increased humidity",
        "higher humidity",
        "enclosure humidity rises",
    ],
    "EFFECT_NOISE_REDUCTION": [
        "noise decreases",
        "noise drops",
        "noise is reduced",
        "noise reduction",
        "quieter",
        "ambient noise decreases",
    ],
    "EFFECT_VIBRATION_REDUCTION": [
        "vibration decreases",
        "vibration drops",
        "vibration is reduced",
        "vibration reduction",
        "reduced vibration",
        "frame vibration decreases",
    ],
    "EFFECT_SIGNAL_RETENTION": [
        "signal is retained",
        "signal remains",
        "signal persists",
        "signal retention",
        "signal is retained longer",
    ],
    "EFFECT_DOOR_UNLOCKS": [
        "door unlocks",
        "door is unlocked",
        "door becomes unlocked",
        "unlock the door",
        "the door unlocks",
    ],
    "EFFECT_BLUE_SIGNAL": [
        "blue signal",
        "emits a blue signal",
        "produces a blue signal",
        "transmits a blue signal",
        "tower emits a blue signal",
    ],
    "EFFECT_WATER_CLARITY": [
        "water becomes clearer",
        "clearer water",
        "water clarity increases",
        "water clarity improves",
        "water becomes more clear",
    ],
}


GENERIC_UNCERTAINTY_PATTERNS = [
    r"\bneed(?:s|ed)? more information\b",
    r"\bneed(?:s|ed)? more context\b",
    r"\bnot enough information\b",
    r"\binsufficient information\b",
    r"\bneed(?:s|ed)? to know the rule\b",
    r"\bneed(?:s|ed)? to know the learned causal rule\b",
    r"\bhave not provided the learned causal rule\b",
    r"\bhaven t provided the learned causal rule\b",
    r"\bhaven t been given the learned causal rule\b",
    r"\bwithout knowing the rule\b",
    r"\bwithout knowing the learned causal rule\b",
    r"\bwithout the learned causal rule\b",
    r"\bcannot determine\b",
    r"\bcan not determine\b",
    r"\bcan t determine\b",
    r"\bcannot give (?:a )?definitive answer\b",
    r"\bcan not give (?:a )?definitive answer\b",
    r"\bcan t give (?:a )?definitive answer\b",
    r"\bimpossible to determine\b",
    r"\bunclear\b",
    r"\bunknown\b",
    r"\bit depends\b",
    r"\bdepends on\b",
]


CAUSAL_REJECTION_PATTERNS = [
    r"^\s*no\b",
    r"\bdoes not prove\b",
    r"\bdoesn t prove\b",
    r"\bdo not prove\b",
    r"\bdon t prove\b",
    r"\bnot necessarily\b",
    r"\bcannot conclude\b",
    r"\bcan not conclude\b",
    r"\bcan t conclude\b",
    r"\bcannot infer\b",
    r"\bcan not infer\b",
    r"\bcan t infer\b",
    r"\bdoes not imply\b",
    r"\bdoesn t imply\b",
    r"\bdo not imply\b",
    r"\bdon t imply\b",
    r"\bdoes not cause\b",
    r"\bdoesn t cause\b",
    r"\bdo not cause\b",
    r"\bdon t cause\b",
    r"\bwould not cause\b",
    r"\bshould not cause\b",
    r"\bno causal effect\b",
    r"\bno causal relationship\b",
    r"\bnot a causal relationship\b",
    r"\bnot prove causation\b",
]


AFFIRMATIVE_POLARITY_PATTERNS = [
    r"^\s*yes\b",
    r"\bshould lead to\b",
    r"\bwould lead to\b",
    r"\bwill lead to\b",
    r"\bleads to\b",
    r"\bshould cause\b",
    r"\bwould cause\b",
    r"\bwill cause\b",
    r"\bcauses\b",
    r"\bis caused by\b",
    r"\bproduces\b",
    r"\bresults in\b",
    r"\bexpected outcome\b",
    r"\bthe outcome is\b",
    r"\boutcome:\s*effect_",
]


def normalize_text(text: Any) -> str:
    text = "" if text is None else str(text)
    text = text.lower().replace("_", " ")
    text = re.sub(r"[^a-z0-9\s]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def extract_labels(text: Any) -> List[str]:
    raw = "" if text is None else str(text).upper()
    out: List[str] = []
    seen: set[str] = set()
    for match in LABEL_RE.finditer(raw):
        label = match.group(0)
        if label not in seen:
            out.append(label)
            seen.add(label)
    return out


def expected_label(example: Mapping[str, Any]) -> str | None:
    scoring = example.get("scoring")
    candidates: List[Any] = [example.get("target"), example.get("exact_answer")]
    if isinstance(scoring, Mapping):
        candidates.append(scoring.get("exact_answer"))
        candidates.extend(scoring.get("required_concepts") or [])
    for candidate in candidates:
        labels = extract_labels(candidate)
        if labels:
            return labels[0]
    return None


def scoring_type(example: Mapping[str, Any]) -> str:
    scoring = example.get("scoring")
    if isinstance(scoring, Mapping) and scoring.get("scoring_type") not in (None, ""):
        return str(scoring["scoring_type"])
    return str(example.get("scoring_type", ""))


def contains_label(label: str | None, response: str) -> bool:
    if not label:
        return False
    return label in extract_labels(response)


def is_uncertain(response: str) -> bool:
    text = normalize_text(response)
    return any(re.search(pattern, text) for pattern in GENERIC_UNCERTAINTY_PATTERNS)


def has_causal_rejection(response: str) -> bool:
    text = normalize_text(response)
    return any(re.search(pattern, text) for pattern in CAUSAL_REJECTION_PATTERNS)


def has_affirmative_polarity(response: str) -> bool:
    text = normalize_text(response)
    return any(re.search(pattern, text) for pattern in AFFIRMATIVE_POLARITY_PATTERNS)


def affirmative_is_embedded_in_no_effect_rejection(response: str) -> bool:
    """Detect causal verbs used only inside a rejected implication claim."""

    text = normalize_text(response)
    if re.match(r"^\s*(yes|correct|true)\b", text):
        return False
    causal_verb = r"(?:cause|causes|caused|lead|leads|led|produce|produces|produced|result|results|resulted)"
    patterns = [
        rf"\bdoes not prove(?: that)?\b.*\b{causal_verb}\b",
        rf"\bdoesn t prove(?: that)?\b.*\b{causal_verb}\b",
        rf"\bdo not prove(?: that)?\b.*\b{causal_verb}\b",
        rf"\bdon t prove(?: that)?\b.*\b{causal_verb}\b",
        rf"\bnot necessarily imply\b.*\b{causal_verb}\b",
        rf"\bdoes not imply(?: that)?\b.*\b{causal_verb}\b",
        rf"\bdoesn t imply(?: that)?\b.*\b{causal_verb}\b",
        rf"\bcannot infer(?: that)?\b.*\b{causal_verb}\b",
        rf"\bcan not infer(?: that)?\b.*\b{causal_verb}\b",
        rf"\bcan t infer(?: that)?\b.*\b{causal_verb}\b",
        rf"\bcannot conclude(?: that)?\b.*\b{causal_verb}\b",
        rf"\bcan not conclude(?: that)?\b.*\b{causal_verb}\b",
        rf"\bcan t conclude(?: that)?\b.*\b{causal_verb}\b",
    ]
    return any(re.search(pattern, text) for pattern in patterns)


def aliases_for_label(label: str) -> List[str]:
    if label == NO_EFFECT_LABEL:
        return [NO_EFFECT_LABEL]
    if label not in EFFECT_ALIASES:
        raise ValueError(
            f"Unknown causal EFFECT label {label!r}; add an explicit alias table entry before rescoring."
        )
    label_without_prefix = label.removeprefix("EFFECT_")
    aliases = [
        label,
        label_without_prefix,
        label_without_prefix.replace("_", " "),
        *EFFECT_ALIASES[label],
    ]
    out: List[str] = []
    seen: set[str] = set()
    for alias in aliases:
        norm = normalize_text(alias)
        if norm and norm not in seen:
            out.append(norm)
            seen.add(norm)
    return out


def alias_matches(response_norm: str, alias: str) -> List[re.Match[str]]:
    return list(re.finditer(rf"(?<!\w){re.escape(alias)}(?!\w)", response_norm))


def alias_match_is_negated(response_norm: str, match: re.Match[str]) -> bool:
    alias = match.group(0)
    prefix = response_norm[: match.start()].strip()
    suffix = response_norm[match.end() :].strip()

    before_patterns = [
        r"(?:^|\s)no$",
        r"(?:^|\s)not(?:\s+(?:a|an|the))?$",
        rf"(?:^|\s)(?:does|do|did|will|would|should|can|could|is|are|was|were)\s+not(?:\s+\w+){{0,4}}$",
        rf"(?:^|\s)(?:doesn|don|didn|won|wouldn|shouldn|can|couldn|isn|aren|wasn|weren)\s+t(?:\s+\w+){{0,4}}$",
        rf"(?:^|\s)without(?:\s+\w+){{0,4}}$",
    ]
    if any(re.search(pattern, prefix) for pattern in before_patterns):
        return True

    after_patterns = [
        r"^(?:is|are|was|were|will|would|should|can|could)\s+not\b",
        r"^(?:isn|aren|wasn|weren|won|wouldn|shouldn|can|couldn)\s+t\b",
    ]
    return any(re.search(pattern, suffix) for pattern in after_patterns)


def effect_asserted(expected: str | None, response: str) -> bool:
    if not expected or expected == NO_EFFECT_LABEL:
        return False
    response_norm = normalize_text(response)
    for alias in aliases_for_label(expected):
        for match in alias_matches(response_norm, alias):
            if not alias_match_is_negated(response_norm, match):
                return True
    return False


@dataclass(frozen=True)
class CausalScore:
    content_correct: bool
    format_correct: bool
    ambiguous: bool
    score_version: str = CAUSAL_SCORE_VERSION
    expected_label: str | None = None
    scoring_type: str | None = None

    def as_fields(self) -> Dict[str, Any]:
        return {
            "causal_content_correct": bool(self.content_correct),
            "causal_format_correct": bool(self.format_correct),
            "causal_ambiguous": bool(self.ambiguous),
            "causal_score_version": self.score_version,
            "causal_content_expected_label": self.expected_label,
            "causal_content_scoring_type": self.scoring_type,
        }


def score_causal_content(
    example: Mapping[str, Any],
    response: str,
    *,
    score_version: str = CAUSAL_SCORE_VERSION,
) -> CausalScore:
    if score_version != CAUSAL_SCORE_VERSION:
        raise ValueError(f"Unsupported causal score version: {score_version}")

    expected = expected_label(example)
    stype = scoring_type(example)
    if expected is None:
        raise ValueError(f"Could not infer causal expected label for example {example.get('example_id')!r}")

    format_correct = contains_label(expected, response)
    uncertain = is_uncertain(response)
    rejection = has_causal_rejection(response)
    affirmative = has_affirmative_polarity(response)
    asserted = effect_asserted(expected, response) if expected != NO_EFFECT_LABEL else False

    content_correct = False

    if stype == "causal_no_effect":
        contradictory = bool(
            rejection
            and affirmative
            and not affirmative_is_embedded_in_no_effect_rejection(response)
        )
        ambiguous = bool(uncertain or contradictory)
        content_correct = bool(format_correct or (rejection and not uncertain and not contradictory))
    elif stype == "causal_positive_polarity":
        ambiguous = bool(uncertain or (affirmative and rejection))
        content_correct = bool(not uncertain and not rejection and (affirmative or asserted))
    elif stype in {"causal_application", "causal_generalization"}:
        ambiguous = bool(uncertain or (affirmative and rejection))
        content_correct = bool(not uncertain and not rejection and asserted)
    else:
        raise ValueError(f"Unsupported causal scoring_type={stype!r} for example {example.get('example_id')!r}")

    if ambiguous:
        content_correct = False

    return CausalScore(
        content_correct=content_correct,
        format_correct=format_correct,
        ambiguous=ambiguous,
        score_version=score_version,
        expected_label=expected,
        scoring_type=stype,
    )


def validate_alias_coverage(examples: Sequence[Mapping[str, Any]]) -> None:
    missing = sorted(
        {
            label
            for example in examples
            for label in [expected_label(example)]
            if label and label != NO_EFFECT_LABEL and label not in EFFECT_ALIASES
        }
    )
    if missing:
        raise ValueError(f"Missing causal aliases for labels: {missing}")
