#!/usr/bin/env python3
"""
Validate the localized-learning dataset.

Checks:
- each prompt example has required fields
- each latent spec has all intended splits
- no duplicate example IDs
- expected split counts per spec
- lexical/factual required latent fields are present
- negative controls have forbidden concepts
"""

from pathlib import Path
import json
from collections import Counter, defaultdict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

REQUIRED_EXAMPLE_FIELDS = {"example_id", "spec_id", "learning_type", "split", "prompt", "target", "scoring", "latent_spec", "metadata"}
EXPECTED_SPLIT_COUNTS = {"train": 12, "id_eval": 5, "paraphrase_eval": 5, "generalization": 6, "negative_control": 6}

def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def validate():
    specs = read_jsonl(DATA_DIR / "latent_specs.jsonl")
    examples = read_jsonl(DATA_DIR / "prompt_examples.jsonl")
    errors = []

    ids = [e["example_id"] for e in examples]
    duplicates = [k for k, v in Counter(ids).items() if v > 1]
    if duplicates:
        errors.append(f"Duplicate example_id values: {duplicates[:5]}")

    spec_ids = {s["spec_id"] for s in specs}
    for e in examples:
        missing = REQUIRED_EXAMPLE_FIELDS - set(e)
        if missing:
            errors.append(f"{e.get('example_id', '<no-id>')} missing fields: {sorted(missing)}")
        if e["spec_id"] not in spec_ids:
            errors.append(f"{e['example_id']} references missing spec {e['spec_id']}")
        if e["split"] not in EXPECTED_SPLIT_COUNTS:
            errors.append(f"{e['example_id']} has unknown split {e['split']}")
        if e["split"] == "negative_control" and not e["scoring"].get("forbidden_concepts"):
            errors.append(f"{e['example_id']} negative control missing forbidden_concepts")

    split_by_spec = defaultdict(Counter)
    for e in examples:
        split_by_spec[e["spec_id"]][e["split"]] += 1

    for sid in spec_ids:
        for split, expected in EXPECTED_SPLIT_COUNTS.items():
            got = split_by_spec[sid][split]
            if got != expected:
                errors.append(f"{sid} split {split}: expected {expected}, got {got}")

    for s in specs:
        ls = s["latent_spec"]
        if s["learning_type"] == "lexical_binding":
            for k in ["novel_term", "target_concept", "category", "attributes", "affordances", "non_examples"]:
                if k not in ls:
                    errors.append(f"{s['spec_id']} missing lexical latent field {k}")
        elif s["learning_type"] == "factual_association":
            for k in ["subject", "relation", "object", "subject_type", "object_type"]:
                if k not in ls:
                    errors.append(f"{s['spec_id']} missing factual latent field {k}")

    print("Dataset validation summary")
    print("--------------------------")
    print(f"latent specs: {len(specs)}")
    print(f"prompt examples: {len(examples)}")
    print("learning types:", Counter(s["learning_type"] for s in specs))
    print("splits:", Counter(e["split"] for e in examples))
    if errors:
        print(f"\nFAILED with {len(errors)} errors")
        for err in errors[:25]:
            print(" -", err)
        raise SystemExit(1)
    print("\nPASSED")

if __name__ == "__main__":
    validate()
