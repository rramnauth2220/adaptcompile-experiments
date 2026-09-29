import csv
import unittest
from collections import Counter, defaultdict
from pathlib import Path

from src.common import read_jsonl
from src.make_calibration_manifest import build_manifest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
FINAL_LOCALIZATION_DIR = ARTIFACTS_DIR / "localization" / "primary" / "cross_objective_localization"

EXPECTED_LEARNING_TYPES = {
    "lexical_binding",
    "factual_association",
    "behavioral_policy",
    "causal_mapping",
    "procedural_reasoning",
}

EXPECTED_SPLIT_COUNTS = {
    "train": 12,
    "id_eval": 5,
    "paraphrase_eval": 5,
    "generalization": 6,
    "negative_control": 6,
}

FINAL_OBJECTIVE_BUDGETS = {
    "Lexical binding": 10,
    "Factual association": 8,
    "Behavioral policy": 10,
    "Causal mapping": 10,
    "Procedural reasoning": 8,
}

FINAL_MANIFEST_RUN_IDS = {
    "calib::lexical_binding::specs25::budget10",
    "calib::factual_association::specs25::budget8",
    "calib::behavioral_policy::specs25::budget10",
    "calib::causal_mapping::specs25::budget10",
    "calib::procedural_reasoning::specs25::budget8",
}

FINAL_EXPECTED_MEANS = {
    ("Lexical binding", "Full"): (0.9973333333333333, 0.8488888888888889, 0.39666666666666667),
    ("Lexical binding", "Early"): (0.972, 0.4055555555555555, 0.68),
    ("Lexical binding", "Middle"): (0.864, 0.5066666666666667, 0.44222222222222224),
    ("Lexical binding", "Late"): (0.7626666666666667, 0.5499999999999999, 0.33666666666666667),
    ("Factual association", "Full"): (0.896, 0.82, 0.3333333333333333),
    ("Factual association", "Early"): (0.324, 0.2044444444444444, 0.3755555555555555),
    ("Factual association", "Middle"): (0.38000000000000006, 0.35333333333333333, 0.4688888888888889),
    ("Factual association", "Late"): (0.5213333333333333, 0.6311111111111112, 0.5711111111111111),
    ("Behavioral policy", "Full"): (0.94, 0.9422222222222222, 0.8955555555555555),
    ("Behavioral policy", "Early"): (0.8413333333333334, 0.8822222222222221, 0.7311111111111112),
    ("Behavioral policy", "Middle"): (0.88, 0.9044444444444445, 0.8577777777777778),
    ("Behavioral policy", "Late"): (0.9306666666666666, 0.8755555555555556, 0.6422222222222222),
    ("Causal mapping", "Full"): (1.0, 0.8022222222222223, 0.5),
    ("Causal mapping", "Early"): (0.9346666666666668, 0.5511111111111111, 0.28444444444444444),
    ("Causal mapping", "Middle"): (0.9853333333333333, 0.68, 0.5),
    ("Causal mapping", "Late"): (0.9626666666666667, 0.5866666666666667, 0.4466666666666667),
    ("Procedural reasoning", "Full"): (0.8266666666666668, 0.6866666666666666, 0.7511111111111112),
    ("Procedural reasoning", "Early"): (0.7493333333333334, 0.3377777777777778, 0.43999999999999995),
    ("Procedural reasoning", "Middle"): (0.7559999999999999, 0.6244444444444445, 0.7644444444444445),
    ("Procedural reasoning", "Late"): (0.6, 0.5866666666666667, 0.82),
}


def read_csv_rows(path: Path):
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def repo_artifact_path(path_text: str) -> Path:
    """Resolve artifact paths stored with either Windows or POSIX separators."""
    normalized = str(path_text).replace("\\", "/")
    path = Path(normalized)
    return path if path.is_absolute() else PROJECT_ROOT / path


def mean(values):
    return sum(values) / len(values)


class DatasetArtifactContractTest(unittest.TestCase):
    def setUp(self):
        self.specs = read_jsonl(DATA_DIR / "latent_specs.jsonl")
        self.examples = read_jsonl(DATA_DIR / "prompt_examples.jsonl")

    def test_dataset_counts_match_current_artifact_contract(self):
        self.assertEqual(len(self.specs), 600)
        self.assertEqual(len(self.examples), 20400)

        self.assertEqual(set(Counter(s["learning_type"] for s in self.specs)), EXPECTED_LEARNING_TYPES)
        self.assertEqual(Counter(s["learning_type"] for s in self.specs), Counter({lt: 120 for lt in EXPECTED_LEARNING_TYPES}))
        self.assertEqual(
            Counter(e["split"] for e in self.examples),
            Counter({
                "train": 7200,
                "id_eval": 3000,
                "paraphrase_eval": 3000,
                "generalization": 3600,
                "negative_control": 3600,
            }),
        )

    def test_dataset_ids_and_per_spec_split_counts_are_consistent(self):
        spec_ids = {s["spec_id"] for s in self.specs}
        example_ids = [e["example_id"] for e in self.examples]
        self.assertEqual(len(example_ids), len(set(example_ids)))

        split_by_spec = defaultdict(Counter)
        for example in self.examples:
            self.assertIn(example["spec_id"], spec_ids)
            self.assertIn(example["learning_type"], EXPECTED_LEARNING_TYPES)
            self.assertIn(example["split"], EXPECTED_SPLIT_COUNTS)
            self.assertTrue(example.get("prompt"))
            self.assertIn("scoring", example)
            self.assertIsInstance(example["scoring"], dict)
            split_by_spec[example["spec_id"]][example["split"]] += 1

        for spec_id in spec_ids:
            self.assertEqual(split_by_spec[spec_id], Counter(EXPECTED_SPLIT_COUNTS), spec_id)

    def test_base_manifest_generator_is_deterministic_and_internally_valid(self):
        first = build_manifest(self.examples)
        second = build_manifest(self.examples)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 100)

        example_ids = {e["example_id"] for e in self.examples}
        for row in first:
            n_specs = int(row["n_specs"])
            budget = int(row["train_budget_per_spec"])
            self.assertEqual(len(row["spec_ids"]), n_specs)
            self.assertEqual(len(row["train_example_ids"]), n_specs * budget)
            self.assertEqual(len(row["id_eval_example_ids"]), n_specs * 5)
            self.assertEqual(len(row["paraphrase_eval_example_ids"]), n_specs * 5)
            self.assertEqual(len(row["generalization_example_ids"]), n_specs * 6)
            self.assertEqual(len(row["negative_control_example_ids"]), n_specs * 6)
            for key in [
                "train_example_ids",
                "id_eval_example_ids",
                "paraphrase_eval_example_ids",
                "generalization_example_ids",
                "negative_control_example_ids",
            ]:
                self.assertTrue(set(row[key]).issubset(example_ids), row["run_id"])


class CalibrationManifestContractTest(unittest.TestCase):
    def setUp(self):
        self.examples = read_jsonl(DATA_DIR / "prompt_examples.jsonl")
        self.manifest = read_jsonl(DATA_DIR / "calibration_manifest.jsonl")
        self.examples_by_id = {row["example_id"]: row for row in self.examples}

    def test_curated_manifest_shape_and_key_final_runs_are_preserved(self):
        self.assertEqual(len(self.manifest), 78)

        run_ids = [row["run_id"] for row in self.manifest]
        self.assertEqual(len(run_ids), len(set(run_ids)))
        self.assertTrue(FINAL_MANIFEST_RUN_IDS.issubset(set(run_ids)))
        self.assertTrue(all("train_budget_per_spec" in row for row in self.manifest))
        self.assertTrue(all("budget" not in row for row in self.manifest))

    def test_manifest_references_existing_examples_with_expected_splits(self):
        split_keys = {
            "train_example_ids": "train",
            "id_eval_example_ids": "id_eval",
            "paraphrase_eval_example_ids": "paraphrase_eval",
            "generalization_example_ids": "generalization",
            "negative_control_example_ids": "negative_control",
        }

        for row in self.manifest:
            run_id = row["run_id"]
            n_specs = int(row["n_specs"])
            budget = int(row["train_budget_per_spec"])

            self.assertEqual(len(row["spec_ids"]), n_specs, run_id)
            self.assertEqual(len(row["train_example_ids"]), n_specs * budget, run_id)
            self.assertEqual(len(row["id_eval_example_ids"]), n_specs * 5, run_id)
            self.assertEqual(len(row["paraphrase_eval_example_ids"]), n_specs * 5, run_id)
            self.assertEqual(len(row["generalization_example_ids"]), n_specs * 6, run_id)
            self.assertEqual(len(row["negative_control_example_ids"]), n_specs * 6, run_id)

            for key, expected_split in split_keys.items():
                ids = row[key]
                self.assertEqual(len(ids), len(set(ids)), f"{run_id} has duplicate ids in {key}")
                for example_id in ids:
                    self.assertIn(example_id, self.examples_by_id, f"{run_id} references missing {example_id}")
                    self.assertEqual(self.examples_by_id[example_id]["split"], expected_split, f"{run_id} {example_id}")


class FinalResultsContractTest(unittest.TestCase):
    def test_final_cross_objective_artifacts_exist_and_are_nonempty(self):
        expected_files = [
            "cross_objective_localization_profile.csv",
            "cross_objective_localization_profile_by_seed.csv",
            "cross_objective_localization_table.tex",
            "cross_objective_localization_table_main.tex",
            "cross_objective_localization_table_main_means.csv",
            "cross_objective_localization_table_by_seed.csv",
            "main_cross_objective_localization_figure_block.tex",
        ]

        for filename in expected_files:
            path = FINAL_LOCALIZATION_DIR / filename
            self.assertTrue(path.exists(), filename)
            self.assertGreater(path.stat().st_size, 100, filename)

        figure_path = ARTIFACTS_DIR / "figures" / "main" / "localization" / "main_cross_objective_localization.pdf"
        self.assertTrue(figure_path.exists())
        self.assertGreater(figure_path.stat().st_size, 100)

    def test_final_main_table_matches_frozen_reported_means(self):
        rows = read_csv_rows(FINAL_LOCALIZATION_DIR / "cross_objective_localization_table_main_means.csv")

        self.assertEqual(len(rows), 20)
        seen = set()
        for row in rows:
            key = (row["objective"], row["condition"])
            seen.add(key)

            self.assertIn(key, FINAL_EXPECTED_MEANS)
            self.assertEqual(int(row["calibrated_budget"]), FINAL_OBJECTIVE_BUDGETS[row["objective"]])
            self.assertEqual(int(row["n_seeds"]), 3)

            expected_acquisition, expected_transfer, expected_boundedness = FINAL_EXPECTED_MEANS[key]
            self.assertAlmostEqual(float(row["acquisition_mean"]), expected_acquisition, places=12)
            self.assertAlmostEqual(float(row["transfer_mean"]), expected_transfer, places=12)
            self.assertAlmostEqual(float(row["boundedness_mean"]), expected_boundedness, places=12)

        self.assertEqual(seen, set(FINAL_EXPECTED_MEANS))

    def test_final_main_table_is_consistent_with_by_seed_table(self):
        main_rows = read_csv_rows(FINAL_LOCALIZATION_DIR / "cross_objective_localization_table_main_means.csv")
        by_seed_rows = read_csv_rows(FINAL_LOCALIZATION_DIR / "cross_objective_localization_table_by_seed.csv")

        self.assertEqual(len(by_seed_rows), 60)

        grouped = defaultdict(list)
        for row in by_seed_rows:
            source_dir = row["source_dir"].replace("\\", "/")
            self.assertTrue(source_dir.startswith("<ARCHIVE_ROOT>/"), source_dir)
            self.assertNotRegex(source_dir, r"^[A-Za-z]:/")
            key = (
                row["objective"],
                int(row["calibrated_budget"]),
                row["localization_condition"].title(),
            )
            grouped[key].append(row)

        for row in main_rows:
            key = (row["objective"], int(row["calibrated_budget"]), row["condition"])
            seed_rows = grouped[key]
            self.assertEqual(len(seed_rows), 3, key)
            self.assertEqual({int(seed_row["seed"]) for seed_row in seed_rows}, {11, 22, 33})

            self.assertAlmostEqual(
                float(row["acquisition_mean"]),
                mean([float(seed_row["acquisition"]) for seed_row in seed_rows]),
                places=12,
            )
            self.assertAlmostEqual(
                float(row["transfer_mean"]),
                mean([float(seed_row["generalization"]) for seed_row in seed_rows]),
                places=12,
            )
            self.assertAlmostEqual(
                float(row["boundedness_mean"]),
                mean([float(seed_row["boundedness"]) for seed_row in seed_rows]),
                places=12,
            )


if __name__ == "__main__":
    unittest.main()
