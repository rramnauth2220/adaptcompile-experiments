#!/usr/bin/env python3
"""Construct the fixed preservation pool for adaptation-compiler experiments."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

try:  # pragma: no cover
    from .compiler_common import read_jsonl, write_jsonl
except ImportError:  # pragma: no cover
    from compiler_common import read_jsonl, write_jsonl


SOURCE = "compiler_preservation_pool"
SCORER = "normalized_exact_match"
RECOGNIZED_SCORERS = {SCORER}
CATEGORIES = [
    "arithmetic_and_comparison",
    "logical_inference",
    "stable_common_knowledge",
    "deterministic_language_operations",
    "constrained_instruction_following",
]
CATEGORY_PREFIX = {
    "arithmetic_and_comparison": "arithmetic",
    "logical_inference": "logic",
    "stable_common_knowledge": "knowledge",
    "deterministic_language_operations": "language",
    "constrained_instruction_following": "instruction",
}
FORBIDDEN_METADATA_KEYS = {"learning_type", "spec_id", "run_id", "episode_id"}
LATENT_SPEC_RE = re.compile(
    r"\b(?:lexical|factual|behavioral|causal|procedural)_\d{4}\b|compiler::",
    re.IGNORECASE,
)


COMMON_KNOWLEDGE_FACTS = [
    ("What is the chemical formula for water? Answer with the exact formula.", "H2O", "chemistry"),
    ("How many sides does a triangle have? Answer with a number.", "3", "geometry"),
    ("How many degrees are in a right angle? Answer with a number.", "90", "geometry"),
    ("Which gas in air is required for human respiration? Answer with one word.", "oxygen", "biology"),
    ("What gas do plants take in for photosynthesis? Answer with two words.", "carbon dioxide", "biology"),
    ("What planet is known as the Red Planet? Answer with one word.", "Mars", "astronomy"),
    ("What is the largest ocean on Earth? Answer with the ocean name.", "Pacific Ocean", "geography"),
    ("What is the smallest prime number? Answer with a number.", "2", "math"),
    ("How many days are in a leap year? Answer with a number.", "366", "calendar"),
    ("Which continent contains Kenya? Answer with the continent name.", "Africa", "geography"),
    ("Which continent contains Brazil? Answer with the continent name.", "South America", "geography"),
    ("What is the freezing point of water in Celsius? Answer with a number.", "0", "science"),
    ("What is the boiling point of water in Celsius at sea level? Answer with a number.", "100", "science"),
    ("What organ pumps blood through the human body? Answer with one word.", "heart", "biology"),
    ("What part of a plant absorbs water from soil? Answer with one word.", "roots", "biology"),
    ("What force pulls objects toward Earth? Answer with one word.", "gravity", "physics"),
    ("What star is at the center of the Solar System? Answer with one word.", "Sun", "astronomy"),
    ("How many minutes are in one hour? Answer with a number.", "60", "time"),
    ("How many hours are in one day? Answer with a number.", "24", "time"),
    ("How many months are in one year? Answer with a number.", "12", "calendar"),
    ("What shape has four equal sides and four right angles? Answer with one word.", "square", "geometry"),
    ("What is the opposite of north on a compass? Answer with one word.", "south", "geography"),
    ("Which ocean is between Africa and Australia? Answer with the ocean name.", "Indian Ocean", "geography"),
    ("What is the largest land animal? Answer with two words.", "African elephant", "biology"),
    ("What is the process by which plants make food using light called? Answer with one word.", "photosynthesis", "biology"),
    ("Which chemical element has the symbol Ca? Answer with one word.", "calcium", "chemistry"),
    ("What is the basic unit of life? Answer with one word.", "cell", "biology"),
    ("What is the center of an atom called? Answer with one word.", "nucleus", "science"),
    ("What is the charge of an electron? Answer with one word.", "negative", "science"),
    ("What is the charge of a proton? Answer with one word.", "positive", "science"),
    ("What is the neutral particle in an atom called? Answer with one word.", "neutron", "science"),
    ("How many legs does an insect have? Answer with a number.", "6", "biology"),
    ("How many wheels does a standard bicycle have? Answer with a number.", "2", "common_objects"),
    ("What color do you get by mixing red and blue paint? Answer with one word.", "purple", "color"),
    ("What color do you get by mixing blue and yellow paint? Answer with one word.", "green", "color"),
    ("What is the written language system of raised dots used by many blind readers called? Answer with one word.", "Braille", "language"),
    ("Which instrument has black and white keys and is played by pressing them? Answer with one word.", "piano", "music"),
    ("What is the name for a baby dog? Answer with one word.", "puppy", "animals"),
    ("What is the name for a baby cat? Answer with one word.", "kitten", "animals"),
    ("How many letters are in the English alphabet? Answer with a number.", "26", "language"),
    ("What punctuation mark usually ends a question? Answer with two words.", "question mark", "language"),
    ("What is the plural of mouse? Answer with one word.", "mice", "language"),
    ("What is the plural of tooth? Answer with one word.", "teeth", "language"),
    ("What is the singular of geese? Answer with one word.", "goose", "language"),
    ("What is the singular of feet? Answer with one word.", "foot", "language"),
    ("What is the first month of the year? Answer with one word.", "January", "calendar"),
    ("What is the last month of the year? Answer with one word.", "December", "calendar"),
    ("What is the third day of a standard week starting Monday? Answer with one word.", "Wednesday", "calendar"),
    ("How many centimeters are in one meter? Answer with a number.", "100", "measurement"),
    ("How many grams are in one kilogram? Answer with a number.", "1000", "measurement"),
    ("Which is larger: a kilometer or a meter? Answer with one word.", "kilometer", "measurement"),
    ("What is the Roman numeral for five? Answer with the numeral.", "V", "symbols"),
    ("What is the Roman numeral for ten? Answer with the numeral.", "X", "symbols"),
    ("What is the name of Earth's natural satellite? Answer with one word.", "Moon", "astronomy"),
    ("Which planet is closest to the Sun? Answer with one word.", "Mercury", "astronomy"),
    ("What is the largest planet in the Solar System? Answer with one word.", "Jupiter", "astronomy"),
    ("What do bees produce from nectar? Answer with one word.", "honey", "biology"),
    ("Which gas is released as a product of photosynthesis? Answer with one word.", "oxygen", "biology"),
    ("What device is used to tell temperature? Answer with one word.", "thermometer", "science"),
    ("What tool is used to measure length with marked units? Answer with one word.", "ruler", "measurement"),
]


def row_id(category: str, index: int) -> str:
    return f"preserve_{CATEGORY_PREFIX[category]}_{index + 1:04d}"


def make_row(
    category: str,
    index: int,
    prompt: str,
    target: Any,
    template_id: str,
    seed: int,
    metadata: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    meta = {
        "source": SOURCE,
        "template_id": template_id,
        "seed": seed,
    }
    if metadata:
        meta.update(metadata)
    return {
        "example_id": row_id(category, index),
        "category": category,
        "prompt": str(prompt).strip(),
        "target": str(target).strip(),
        "scorer": SCORER,
        "metadata": meta,
    }


def arithmetic_examples(num_per_category: int, seed: int) -> List[Dict[str, Any]]:
    rows = []
    for i in range(num_per_category):
        t = i % 6
        variant = i // 6
        if t == 0:
            a = 11 + variant * 7
            b = 5 + variant * 4
            prompt = f"What is {a} + {b}? Answer with the exact number."
            target = a + b
            template = "addition"
            meta = {"operation": "+", "operands": [a, b]}
        elif t == 1:
            b = 3 + variant * 5
            a = b + 20 + variant * 3
            prompt = f"What is {a} - {b}? Answer with the exact number."
            target = a - b
            template = "subtraction"
            meta = {"operation": "-", "operands": [a, b]}
        elif t == 2:
            a = 2 + variant
            b = 3 + variant * 2
            prompt = f"What is {a} x {b}? Answer with the exact number."
            target = a * b
            template = "multiplication"
            meta = {"operation": "x", "operands": [a, b]}
        elif t == 3:
            a = 20 + variant * 9
            b = 15 + variant * 11
            if a == b:
                b += 1
            prompt = f"Which number is larger: {a} or {b}? Answer with only the number."
            target = max(a, b)
            template = "larger_number"
            meta = {"operation": "max", "operands": [a, b]}
        elif t == 4:
            pct = [10, 20, 25, 50, 75][variant % 5]
            base = 40 * (variant + 2)
            prompt = f"What is {pct}% of {base}? Answer with the exact number."
            target = base * pct // 100
            template = "percentage"
            meta = {"operation": "percent", "percent": pct, "base": base}
        else:
            denom = [2, 3, 4, 5, 6][variant % 5]
            multiplier = 7 + variant * 2
            base = denom * multiplier
            prompt = f"What is 1/{denom} of {base}? Answer with the exact number."
            target = base // denom
            template = "unit_fraction"
            meta = {"operation": "fraction", "numerator": 1, "denominator": denom, "base": base}
        rows.append(make_row("arithmetic_and_comparison", i, prompt, target, template, seed, meta))
    return rows


def logical_examples(num_per_category: int, seed: int) -> List[Dict[str, Any]]:
    names = ["Rilo", "Mava", "Tirin", "Sola", "Nemi", "Faro", "Kiva", "Luro", "Pela", "Daxin"]
    groups = ["daxes", "mips", "lorbs", "nims", "tavos", "zerns", "pelks", "rinns", "sarels", "vombs"]
    rows = []
    for i in range(num_per_category):
        t = i % 6
        variant = i // 6
        name = names[variant % len(names)]
        group_a = groups[(variant + t) % len(groups)]
        group_b = groups[(variant + t + 3) % len(groups)]
        if t == 0:
            prompt = (
                f"All {group_a} are {group_b}. {name} is a {group_a[:-1]}. "
                f"Is {name} a {group_b[:-1]}? Answer YES or NO."
            )
            target = "YES"
            template = "syllogism_positive"
        elif t == 1:
            prompt = (
                f"No {group_a} are {group_b}. {name} is a {group_a[:-1]}. "
                f"Is {name} a {group_b[:-1]}? Answer YES or NO."
            )
            target = "NO"
            template = "syllogism_negative"
        elif t == 2:
            prompt = (
                f"Every object in Box A is also in Box B. {name} is in Box A. "
                f"Is {name} in Box B? Answer YES or NO."
            )
            target = "YES"
            template = "set_inclusion"
        elif t == 3:
            prompt = (
                f"In scenario {variant + 1}, Statement P is true and Statement Q is false. Are both P and Q true? "
                f"Answer YES or NO."
            )
            target = "NO"
            template = "conjunction"
        elif t == 4:
            prompt = (
                f"Exactly one of Lamp A{variant + 1} and Lamp B{variant + 1} is on. "
                f"Lamp A{variant + 1} is off. Is Lamp B{variant + 1} on? Answer YES or NO."
            )
            target = "YES"
            template = "exclusive_or"
        else:
            prompt = (
                f"Can both statements be true at the same time: '{name} is inside the box' "
                f"and '{name} is not inside the box'? Answer YES or NO."
            )
            target = "NO"
            template = "contradiction"
        rows.append(make_row("logical_inference", i, prompt, target, template, seed))
    return rows


def stable_common_knowledge_examples(num_per_category: int, seed: int) -> List[Dict[str, Any]]:
    if num_per_category > len(COMMON_KNOWLEDGE_FACTS):
        raise ValueError(
            f"Requested {num_per_category} stable-common-knowledge examples, "
            f"but only {len(COMMON_KNOWLEDGE_FACTS)} curated unique facts are available."
        )
    rows = []
    for i in range(num_per_category):
        prompt, target, template = COMMON_KNOWLEDGE_FACTS[i]
        rows.append(
            make_row(
                "stable_common_knowledge",
                i,
                prompt,
                target,
                f"fact_{template}",
                seed,
                {"fact_index": i % len(COMMON_KNOWLEDGE_FACTS)},
            )
        )
    return rows


def language_examples(num_per_category: int, seed: int) -> List[Dict[str, Any]]:
    word_sets = [
        ["silver", "maple", "river"],
        ["orbit", "anchor", "lantern"],
        ["violet", "cedar", "button"],
        ["marble", "cotton", "ember"],
        ["guitar", "forest", "needle"],
        ["planet", "basket", "window"],
        ["candle", "meadow", "thread"],
        ["copper", "garden", "pencil"],
        ["velvet", "harbor", "ladder"],
        ["magnet", "willow", "ribbon"],
    ]
    singular_plural = [
        ("leaf", "singular"),
        ("leaves", "plural"),
        ("child", "singular"),
        ("children", "plural"),
        ("box", "singular"),
        ("boxes", "plural"),
        ("knife", "singular"),
        ("knives", "plural"),
        ("story", "singular"),
        ("stories", "plural"),
    ]
    rows = []
    for i in range(num_per_category):
        t = i % 6
        variant = i // 6
        words = word_sets[variant % len(word_sets)]
        if t == 0:
            phrase = f"{words[0]} {words[1]} {words[2]}"
            prompt = f"How many words are in this phrase: '{phrase}'? Answer with a number."
            target = len(phrase.split())
            template = "word_count"
        elif t == 1:
            word = words[0]
            prompt = f"Write the word '{word}' in uppercase. Answer with only the transformed word."
            target = word.upper()
            template = "uppercase"
        elif t == 2:
            word = words[1].upper()
            prompt = f"Write the word '{word}' in lowercase. Answer with only the transformed word."
            target = word.lower()
            template = "lowercase"
        elif t == 3:
            ordered = sorted(words)
            prompt = (
                f"Alphabetize these words: {words[0]}, {words[1]}, {words[2]}. "
                f"Answer as comma-separated words."
            )
            target = ", ".join(ordered)
            template = "alphabetize"
        elif t == 4:
            word, label = singular_plural[variant % len(singular_plural)]
            prompt = f"Is the word '{word}' singular or plural? Answer SINGULAR or PLURAL."
            target = label.upper()
            template = "singular_plural"
        else:
            word = words[2]
            letter = word[1]
            prompt = f"In the word '{word}', what is the second letter? Answer with one letter."
            target = letter
            template = "letter_position"
        rows.append(make_row("deterministic_language_operations", i, prompt, target, template, seed))
    return rows


def constrained_instruction_examples(num_per_category: int, seed: int) -> List[Dict[str, Any]]:
    item_lists = [
        ["red", "blue", "green"],
        ["oak", "pine", "birch"],
        ["circle", "square", "triangle"],
        ["north", "east", "west"],
        ["copper", "silver", "gold"],
        ["small", "medium", "large"],
        ["spring", "summer", "winter"],
        ["alpha", "beta", "gamma"],
        ["left", "center", "right"],
        ["Monday", "Tuesday", "Friday"],
        ["rose", "tulip", "lily"],
        ["paper", "glass", "metal"],
    ]
    rows = []
    for i in range(num_per_category):
        t = i % 5
        variant = i // 5
        items = item_lists[variant % len(item_lists)]
        if t == 0:
            position = variant % 3
            labels = ["first", "second", "third"]
            prompt = (
                f"Return only the {labels[position]} item from this list: "
                f"{items[0]}, {items[1]}, {items[2]}."
            )
            target = items[position]
            template = "select_list_item"
        elif t == 1:
            nums = [9 + (i % 7), 2 + (i * 3) % 11, 15 + (i * 5) % 13]
            prompt = f"Sort these numbers in ascending order: {nums[0]}, {nums[1]}, {nums[2]}. Use comma-space format."
            target = ", ".join(str(x) for x in sorted(nums))
            template = "sort_numbers"
        elif t == 2:
            value = 10 + (i * 4) % 31
            threshold = 20
            prompt = f"Return only YES or NO: is {value} greater than {threshold}?"
            target = "YES" if value > threshold else "NO"
            template = "yes_no_condition"
        elif t == 3:
            name = ["Mira", "Jon", "Lea", "Omar", "Tess"][i % 5]
            code = f"K{200 + i}"
            prompt = f"Record: name={name}; code={code}; status=ready. Return only the code field."
            target = code
            template = "extract_field"
        else:
            left = ["AB", "CD", "EF", "GH", "JK"][i % 5]
            right = 10 + i
            prompt = f"Format the pair with a colon: left={left}, right={right}. Return only the formatted pair."
            target = f"{left}:{right}"
            template = "deterministic_format"
        rows.append(make_row("constrained_instruction_following", i, prompt, target, template, seed))
    return rows


GENERATOR_BY_CATEGORY = {
    "arithmetic_and_comparison": arithmetic_examples,
    "logical_inference": logical_examples,
    "stable_common_knowledge": stable_common_knowledge_examples,
    "deterministic_language_operations": language_examples,
    "constrained_instruction_following": constrained_instruction_examples,
}


def build_preservation_dataset(num_per_category: int = 60, seed: int = 2026) -> List[Dict[str, Any]]:
    if num_per_category <= 0:
        raise ValueError("num_per_category must be positive.")
    # ``seed`` is recorded as provenance only. The canonical pool is intentionally
    # index-driven so that changing a CLI seed cannot silently change the benchmark.
    rows: List[Dict[str, Any]] = []
    for category in CATEGORIES:
        rows.extend(GENERATOR_BY_CATEGORY[category](num_per_category, seed))
    return rows


def prompt_overlap_with_adaptation(rows: Sequence[Dict[str, Any]], prompt_examples_path: str | Path) -> set[str]:
    path = Path(prompt_examples_path)
    if not path.exists():
        return set()
    adaptation_prompts = {str(row.get("prompt", "")) for row in read_jsonl(path)}
    return {str(row["prompt"]) for row in rows if str(row["prompt"]) in adaptation_prompts}


def validate_dataset(
    rows: Sequence[Dict[str, Any]],
    num_per_category: int,
    prompt_examples_path: str | Path = "data/prompt_examples.jsonl",
) -> Dict[str, Any]:
    counts = Counter(row.get("category") for row in rows)
    expected_counts = {category: num_per_category for category in CATEGORIES}
    if dict(counts) != expected_counts:
        raise ValueError(f"Unexpected category counts: {dict(counts)} != {expected_counts}")

    ids = [row.get("example_id") for row in rows]
    if len(set(ids)) != len(ids):
        duplicates = [item for item, count in Counter(ids).items() if count > 1]
        raise ValueError(f"Duplicate example_id values: {duplicates[:5]}")

    prompt_targets = [(row.get("prompt"), row.get("target")) for row in rows]
    if len(set(prompt_targets)) != len(prompt_targets):
        duplicates = [item for item, count in Counter(prompt_targets).items() if count > 1]
        raise ValueError(f"Duplicate prompt-target pairs: {duplicates[:3]}")

    for row in rows:
        for field in ["example_id", "category", "prompt", "target", "scorer", "metadata"]:
            if row.get(field) in (None, ""):
                raise ValueError(f"{row.get('example_id')} has empty required field {field!r}")
        if row["category"] not in CATEGORIES:
            raise ValueError(f"{row['example_id']} has unknown category {row['category']!r}")
        if row["scorer"] not in RECOGNIZED_SCORERS:
            raise ValueError(f"{row['example_id']} has unknown scorer {row['scorer']!r}")
        metadata = row.get("metadata")
        if not isinstance(metadata, dict):
            raise ValueError(f"{row['example_id']} metadata must be an object")
        if metadata.get("source") != SOURCE:
            raise ValueError(f"{row['example_id']} metadata.source must be {SOURCE!r}")
        if "template_id" not in metadata or "seed" not in metadata:
            raise ValueError(f"{row['example_id']} metadata missing template_id or seed")
        bad_top = FORBIDDEN_METADATA_KEYS & set(row)
        bad_meta = FORBIDDEN_METADATA_KEYS & set(metadata)
        if bad_top or bad_meta:
            raise ValueError(f"{row['example_id']} contains adaptation metadata keys: {sorted(bad_top | bad_meta)}")
        serialized = json.dumps(row, sort_keys=True)
        if LATENT_SPEC_RE.search(serialized):
            raise ValueError(f"{row['example_id']} appears to reference an adaptation latent spec id")

    overlaps = prompt_overlap_with_adaptation(rows, prompt_examples_path)
    if overlaps:
        sample = sorted(overlaps)[:3]
        raise ValueError(f"Preservation prompts overlap adaptation prompts exactly: {sample}")

    return {
        "total_examples": len(rows),
        "category_counts": dict(counts),
        "scorer_counts": dict(Counter(row["scorer"] for row in rows)),
        "duplicate_prompt_target_pairs": 0,
        "adaptation_prompt_overlap": len(overlaps),
    }


def print_stats(stats: Dict[str, Any]) -> None:
    print(f"Total examples: {stats['total_examples']}")
    print("Examples/category:")
    for category in CATEGORIES:
        print(f"  {category}: {stats['category_counts'].get(category, 0)}")
    print("Scorer distribution:")
    for scorer, count in sorted(stats["scorer_counts"].items()):
        print(f"  {scorer}: {count}")
    print(f"Duplicate prompt-target pairs: {stats['duplicate_prompt_target_pairs']}")
    print(f"Exact prompt overlap with data/prompt_examples.jsonl: {stats['adaptation_prompt_overlap']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the adaptation-compiler preservation dataset.")
    parser.add_argument("--num_per_category", type=int, default=60)
    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
        help="Provenance seed recorded in metadata; the canonical pool itself is index-driven.",
    )
    parser.add_argument("--output", default="data/compiler/preservation_examples.jsonl")
    parser.add_argument("--prompt_examples_path", default="data/prompt_examples.jsonl")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = build_preservation_dataset(num_per_category=args.num_per_category, seed=args.seed)
    stats = validate_dataset(
        rows,
        num_per_category=args.num_per_category,
        prompt_examples_path=args.prompt_examples_path,
    )
    write_jsonl(args.output, rows)
    print(f"Wrote preservation dataset to {args.output}")
    print_stats(stats)


if __name__ == "__main__":
    main()
