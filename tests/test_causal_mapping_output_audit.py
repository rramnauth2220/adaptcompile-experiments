import csv
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "audits" / "audit_causal_mapping_outputs.py"


def load_module():
    spec = importlib.util.spec_from_file_location("audit_causal_mapping_outputs", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


class CausalMappingOutputAuditTest(unittest.TestCase):
    def test_main_counts_parsed_no_effect_on_paraphrase_positive(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_path = root / "causal_mapping_result.jsonl"
            output_path = root / "audit.csv"
            write_jsonl(
                input_path,
                [
                    {
                        "learning_type": "causal_mapping",
                        "split": "paraphrase_eval",
                        "causal_expected_label": "EFFECT_BLUE_SIGNAL",
                        "causal_predicted_labels": ["NO_CAUSAL_EFFECT"],
                        "strict_correct": 0,
                    },
                    {
                        "learning_type": "causal_mapping",
                        "split": "generalization",
                        "target": "Outcome: EFFECT_BLUE_SIGNAL.",
                        "response": "Outcome: NO_CAUSAL_EFFECT. The learned causal rule does not apply.",
                        "strict_correct": 0,
                    },
                ],
            )

            old_argv = sys.argv
            sys.argv = [
                "audit_causal_mapping_outputs.py",
                "--inputs",
                str(input_path),
                "--output_csv",
                str(output_path),
            ]
            try:
                module.main()
            finally:
                sys.argv = old_argv

            with open(output_path, encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))

        by_split = {row["split"]: row for row in rows}
        self.assertEqual(by_split["paraphrase_eval"]["no_effect_prediction_rate"], "1.0")
        self.assertEqual(by_split["generalization"]["no_effect_prediction_rate"], "1.0")
        self.assertEqual(by_split["paraphrase_eval"]["first_label_no_effect_rate"], "1.0")


if __name__ == "__main__":
    unittest.main()
