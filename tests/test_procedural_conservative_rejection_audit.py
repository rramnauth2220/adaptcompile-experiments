import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "audits" / "audit_procedural_conservative_rejection.py"


def load_module():
    spec = importlib.util.spec_from_file_location("audit_procedural_conservative_rejection", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


class ProceduralConservativeRejectionAuditTest(unittest.TestCase):
    def test_counts_parsed_incomplete_prediction_on_paraphrase_positive(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "procedural_reasoning_result.jsonl"
            write_jsonl(
                path,
                [
                    {
                        "learning_type": "procedural_reasoning",
                        "split": "paraphrase_eval",
                        "target": "Outcome: OUTCOME_FILTER_STABILIZED.",
                        "procedural_predicted_label": "PROCEDURE_INCOMPLETE",
                    }
                ],
            )

            result = module.audit_file(path)

        self.assertIsNotNone(result)
        self.assertEqual(result["positive_examples"], 1)
        self.assertEqual(result["procedure_incomplete_on_positive"], 1)
        self.assertEqual(result["paraphrase_eval_rate"], 1.0)

    def test_counts_incomplete_inside_raw_response_text(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "procedural_reasoning_result.jsonl"
            write_jsonl(
                path,
                [
                    {
                        "learning_type": "procedural_reasoning",
                        "split": "generalization",
                        "target": "Outcome: OUTCOME_PRESSURE_READY.",
                        "response": "Outcome: PROCEDURE_INCOMPLETE. The ordered procedure was not completed.",
                    }
                ],
            )

            result = module.audit_file(path)

        self.assertEqual(result["positive_examples"], 1)
        self.assertEqual(result["procedure_incomplete_on_positive"], 1)

    def test_skips_negative_controls_and_true_incomplete_targets(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "procedural_reasoning_result.jsonl"
            write_jsonl(
                path,
                [
                    {
                        "learning_type": "procedural_reasoning",
                        "split": "negative_control",
                        "target": "Outcome: PROCEDURE_INCOMPLETE.",
                        "procedural_predicted_label": "PROCEDURE_INCOMPLETE",
                    },
                    {
                        "learning_type": "procedural_reasoning",
                        "split": "id_eval",
                        "target": "Outcome: PROCEDURE_INCOMPLETE.",
                        "procedural_predicted_label": "PROCEDURE_INCOMPLETE",
                    },
                    {
                        "learning_type": "procedural_reasoning",
                        "split": "id_eval",
                        "target": "Outcome: OUTCOME_FILTER_STABILIZED.",
                        "procedural_predicted_label": "OUTCOME_FILTER_STABILIZED",
                    },
                ],
            )

            result = module.audit_file(path)

        self.assertEqual(result["positive_examples"], 1)
        self.assertEqual(result["procedure_incomplete_on_positive"], 0)


if __name__ == "__main__":
    unittest.main()
