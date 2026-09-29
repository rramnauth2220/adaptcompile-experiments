import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "audits" / "audit_behavioral_conservative_rejection.py"


def load_module():
    spec = importlib.util.spec_from_file_location("audit_behavioral_conservative_rejection", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


class BehavioralConservativeRejectionAuditTest(unittest.TestCase):
    def test_counts_parsed_no_policy_trigger_on_positive(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "behavioral_policy_result.jsonl"
            write_jsonl(
                path,
                [
                    {
                        "learning_type": "behavioral_policy",
                        "split": "paraphrase_eval",
                        "target": "Action: ASK_FOR_PREFERENCE.",
                        "behavioral_predicted_label": "NO_POLICY_TRIGGER",
                    }
                ],
            )

            result = module.audit_file(path)

        self.assertIsNotNone(result)
        self.assertEqual(result["positive_examples"], 1)
        self.assertEqual(result["no_policy_trigger_on_positive"], 1)
        self.assertEqual(result["paraphrase_eval_rate"], 1.0)

    def test_counts_no_policy_trigger_inside_raw_response(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "behavioral_policy_result.jsonl"
            write_jsonl(
                path,
                [
                    {
                        "learning_type": "behavioral_policy",
                        "split": "generalization",
                        "target": "Action: ASK_FOR_BUDGET.",
                        "response": "Action: NO_POLICY_TRIGGER. The learned policy does not apply.",
                    }
                ],
            )

            result = module.audit_file(path)

        self.assertEqual(result["positive_examples"], 1)
        self.assertEqual(result["no_policy_trigger_on_positive"], 1)

    def test_skips_negative_controls_and_true_no_trigger_targets(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "behavioral_policy_result.jsonl"
            write_jsonl(
                path,
                [
                    {
                        "learning_type": "behavioral_policy",
                        "split": "negative_control",
                        "target": "Action: NO_POLICY_TRIGGER.",
                        "behavioral_predicted_label": "NO_POLICY_TRIGGER",
                    },
                    {
                        "learning_type": "behavioral_policy",
                        "split": "id_eval",
                        "target": "Action: NO_POLICY_TRIGGER.",
                        "behavioral_predicted_label": "NO_POLICY_TRIGGER",
                    },
                    {
                        "learning_type": "behavioral_policy",
                        "split": "id_eval",
                        "target": "Action: ASK_FOR_TIME.",
                        "behavioral_predicted_label": "ASK_FOR_TIME",
                    },
                ],
            )

            result = module.audit_file(path)

        self.assertEqual(result["positive_examples"], 1)
        self.assertEqual(result["no_policy_trigger_on_positive"], 0)


if __name__ == "__main__":
    unittest.main()
