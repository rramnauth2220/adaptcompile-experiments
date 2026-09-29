import tempfile
import unittest
from pathlib import Path

from src.common import read_jsonl, score_response, write_jsonl


class JsonlHelpersTest(unittest.TestCase):
    def test_round_trip_jsonl_creates_parent_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nested" / "rows.jsonl"
            rows = [{"id": "a", "value": 1}, {"id": "b", "value": ["x", "y"]}]

            write_jsonl(path, rows)

            self.assertEqual(read_jsonl(path), rows)


class ScoreResponseTest(unittest.TestCase):
    def test_positive_yes_no_requires_target_mention_for_strict_score(self):
        example = {
            "prompt": "Does a daxon refer to a bicycle?",
            "scoring": {
                "exact_answer": "bicycle",
                "required_concepts": ["bicycle"],
                "scoring_type": "positive_polarity",
            },
        }

        loose_only = score_response(example, "Yes, it does.")
        strict = score_response(example, "Yes, it is a bicycle.")

        self.assertTrue(loose_only["loose_score"])
        self.assertFalse(loose_only["strict_score"])
        self.assertTrue(strict["passed"])

    def test_negative_control_accepts_rejection_even_when_distractor_is_named(self):
        example = {
            "prompt": "Does a daxon refer to a refrigerator?",
            "scoring": {
                "required_concepts": ["reject distractor"],
                "forbidden_concepts": ["refrigerator"],
                "scoring_type": "negative_control",
            },
        }

        scored = score_response(example, "No, a daxon is not a refrigerator.")

        self.assertTrue(scored["passed"])
        self.assertTrue(scored["starts_with_rejection"])

    def test_open_recall_rejects_negated_exact_answer(self):
        example = {
            "prompt": "What is a daxon?",
            "scoring": {
                "exact_answer": "bicycle",
                "required_concepts": ["bicycle"],
                "scoring_type": "retrieval",
            },
        }

        scored = score_response(example, "It is not a bicycle.")

        self.assertFalse(scored["passed"])
        self.assertTrue(scored["contains_exact"])
        self.assertTrue(scored["has_rejection"])


if __name__ == "__main__":
    unittest.main()
