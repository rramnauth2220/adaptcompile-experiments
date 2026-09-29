import argparse
import csv
import tempfile
import unittest
from pathlib import Path

from src.common import write_jsonl
from src.summarize_results import infer_run_metadata, write_summary_csv


class SummarizeResultsTest(unittest.TestCase):
    def test_infers_run_metadata_from_calibration_filename(self):
        metadata = infer_run_metadata("adapter_calib_lexical_binding_specs25_budget8.jsonl")

        self.assertEqual(metadata["condition"], "adapter")
        self.assertEqual(metadata["filename_learning_type"], "lexical_binding")
        self.assertEqual(metadata["n_specs"], 25)
        self.assertEqual(metadata["budget"], 8)

    def test_writes_summary_with_all_modes_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_path = tmp_path / "adapter_calib_lexical_binding_specs2_budget1.jsonl"
            output_path = tmp_path / "summary.csv"
            write_jsonl(
                input_path,
                [
                    {
                        "learning_type": "lexical_binding",
                        "split": "id_eval",
                        "example_mode": "retrieval",
                        "scoring_type": "retrieval",
                        "loose_score": True,
                        "strict_score": True,
                        "passed": True,
                        "target_mentioned": True,
                        "contains_exact": True,
                    },
                    {
                        "learning_type": "lexical_binding",
                        "split": "id_eval",
                        "example_mode": "retrieval",
                        "scoring_type": "retrieval",
                        "loose_score": True,
                        "strict_score": False,
                        "passed": False,
                        "target_mentioned": False,
                    },
                ],
            )
            args = argparse.Namespace(
                inputs=[str(input_path)],
                output_csv=str(output_path),
                no_all_modes=False,
            )

            write_summary_csv(args)

            with open(output_path, encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))

        rows_by_mode = {row["example_mode"]: row for row in rows}
        self.assertEqual(set(rows_by_mode), {"retrieval", "ALL_MODES"})
        self.assertEqual(rows_by_mode["retrieval"]["n"], "2")
        self.assertEqual(rows_by_mode["retrieval"]["condition"], "adapter")
        self.assertEqual(rows_by_mode["retrieval"]["strict_correct"], "1")
        self.assertEqual(float(rows_by_mode["retrieval"]["strict_accuracy"]), 0.5)
        self.assertEqual(float(rows_by_mode["ALL_MODES"]["target_mentioned_rate"]), 0.5)


if __name__ == "__main__":
    unittest.main()
