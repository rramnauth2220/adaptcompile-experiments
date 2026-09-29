import copy
import unittest
from collections import Counter
from pathlib import Path

from src.make_preservation_dataset import (
    CATEGORIES,
    build_preservation_dataset,
    prompt_overlap_with_adaptation,
    validate_dataset,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROMPT_EXAMPLES = PROJECT_ROOT / "data" / "prompt_examples.jsonl"


class PreservationDatasetTest(unittest.TestCase):
    def test_generation_is_deterministic_balanced_and_schema_valid(self):
        first = build_preservation_dataset(num_per_category=60, seed=2026)
        second = build_preservation_dataset(num_per_category=60, seed=2026)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 300)
        self.assertEqual(Counter(row["category"] for row in first), Counter({category: 60 for category in CATEGORIES}))

        stats = validate_dataset(first, num_per_category=60, prompt_examples_path=PROMPT_EXAMPLES)
        self.assertEqual(stats["total_examples"], 300)
        self.assertEqual(stats["duplicate_prompt_target_pairs"], 0)
        self.assertEqual(stats["adaptation_prompt_overlap"], 0)

        for row in first:
            self.assertTrue(row["example_id"].startswith("preserve_"))
            self.assertEqual(row["scorer"], "normalized_exact_match")
            self.assertEqual(row["metadata"]["source"], "compiler_preservation_pool")
            self.assertIn("template_id", row["metadata"])
            self.assertEqual(row["metadata"]["seed"], 2026)
            self.assertNotIn("learning_type", row)
            self.assertNotIn("spec_id", row)
            self.assertNotIn("learning_type", row["metadata"])
            self.assertNotIn("spec_id", row["metadata"])

    def test_seed_is_recorded_and_generation_changes_metadata(self):
        seed_a = build_preservation_dataset(num_per_category=3, seed=2026)
        seed_b = build_preservation_dataset(num_per_category=3, seed=99)

        self.assertNotEqual(seed_a, seed_b)
        self.assertTrue(all(row["metadata"]["seed"] == 2026 for row in seed_a))
        self.assertTrue(all(row["metadata"]["seed"] == 99 for row in seed_b))

    def test_arithmetic_targets_match_metadata(self):
        rows = [
            row
            for row in build_preservation_dataset(num_per_category=60, seed=2026)
            if row["category"] == "arithmetic_and_comparison"
        ]

        for row in rows:
            meta = row["metadata"]
            target = int(row["target"])
            if meta["operation"] == "+":
                self.assertEqual(target, sum(meta["operands"]))
            elif meta["operation"] == "-":
                self.assertEqual(target, meta["operands"][0] - meta["operands"][1])
            elif meta["operation"] == "x":
                self.assertEqual(target, meta["operands"][0] * meta["operands"][1])
            elif meta["operation"] == "max":
                self.assertEqual(target, max(meta["operands"]))
            elif meta["operation"] == "percent":
                self.assertEqual(target, meta["base"] * meta["percent"] // 100)
            elif meta["operation"] == "fraction":
                self.assertEqual(target, meta["base"] // meta["denominator"])
            else:
                self.fail(f"Unexpected arithmetic operation {meta['operation']}")

    def test_logic_targets_match_template(self):
        expected = {
            "syllogism_positive": "YES",
            "syllogism_negative": "NO",
            "set_inclusion": "YES",
            "conjunction": "NO",
            "exclusive_or": "YES",
            "contradiction": "NO",
        }
        rows = [
            row
            for row in build_preservation_dataset(num_per_category=60, seed=2026)
            if row["category"] == "logical_inference"
        ]

        for row in rows:
            template = row["metadata"]["template_id"]
            self.assertEqual(row["target"], expected[template])

    def test_prompts_do_not_exactly_overlap_adaptation_prompts(self):
        rows = build_preservation_dataset(num_per_category=60, seed=2026)

        self.assertEqual(prompt_overlap_with_adaptation(rows, PROMPT_EXAMPLES), set())

    def test_schema_validation_rejects_adaptation_metadata(self):
        rows = build_preservation_dataset(num_per_category=6, seed=2026)
        malformed = copy.deepcopy(rows)
        malformed[0]["metadata"]["spec_id"] = "lexical_0000"

        with self.assertRaisesRegex(ValueError, "adaptation metadata"):
            validate_dataset(malformed, num_per_category=6, prompt_examples_path=PROMPT_EXAMPLES)


if __name__ == "__main__":
    unittest.main()
