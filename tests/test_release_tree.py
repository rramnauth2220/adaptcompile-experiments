import csv
import hashlib
import importlib.util
import json
import re
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = PROJECT_ROOT / "artifacts"
MANIFEST = ARTIFACTS / "MANIFEST.csv"
LLAMA_TABLE_SCRIPT = PROJECT_ROOT / "scripts/reproduce/compiler/make_llama_calibration_table.py"

LOFO_FAMILIES = {
    "behavioral_policy",
    "causal_mapping",
    "factual_association",
    "lexical_binding",
    "procedural_reasoning",
}
ABLATION_FAMILIES = {"episode", "frozen_behavior", "module_probes"}

TEXT_SUFFIXES = {".csv", ".json", ".jsonl", ".md", ".tex", ".txt"}
ABSOLUTE_PATH_PATTERNS = (
    re.compile(r"[A-Za-z]:[\\/]"),
    re.compile(r"/(?:Users|home)/[^/\s]+/"),
    re.compile(r"(?:Desktop|Documents)[\\/]"),
)


def csv_row_count(path: Path) -> int:
    with path.open(encoding="utf-8", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class ReleaseTreeContractTest(unittest.TestCase):
    def test_required_artifact_directories_exist(self):
        required = [
            "llama/geometry/headroom",
            "llama/geometry/robustness",
            "llama/calibration",
            "llama/prediction",
            "llama/selection/utility_specs",
            "llama/ablations",
            "llama/lofo",
            "gemma/calibration",
            "gemma/geometry/headroom",
            "gemma/prediction",
            "gemma/selection",
            "localization/primary",
            "localization/cross_model",
            "localization/parameter_matched",
            "figures/data/llama",
            "figures/data/gemma",
            "figures/data/localization",
            "figures/data/synthesis",
            "figures/main/llama",
            "figures/main/gemma",
            "figures/main/localization",
            "figures/main/synthesis",
            "figures/appendix/llama",
            "figures/appendix/localization",
        ]
        for relative in required:
            self.assertTrue((ARTIFACTS / relative).is_dir(), relative)

    def test_lofo_and_ablation_families_are_complete(self):
        lofo = {path.name for path in (ARTIFACTS / "llama/lofo").iterdir() if path.is_dir()}
        ablations = {path.name for path in (ARTIFACTS / "llama/ablations").iterdir() if path.is_dir()}
        self.assertEqual(lofo, LOFO_FAMILIES)
        self.assertEqual(ablations, ABLATION_FAMILIES)

    def test_primary_structural_row_counts(self):
        expected = {
            "llama/geometry/geometry_records_by_seed.csv": 3200,
            "llama/geometry/geometry_dataset.csv": 2400,
            "llama/prediction/predictions_test.csv": 400,
            "llama/selection/compiler_evaluation.csv": 100,
            "gemma/geometry/geometry_records_by_seed.csv": 3200,
            "gemma/geometry/geometry_dataset.csv": 2400,
            "gemma/prediction/predictions_test.csv": 400,
            "gemma/selection/compiler_evaluation.csv": 100,
        }
        for relative, count in expected.items():
            self.assertEqual(csv_row_count(ARTIFACTS / relative), count, relative)

    def test_gemma_causal_provenance_is_current(self):
        manifest = ARTIFACTS / "gemma/geometry/causal_rescore_manifest.csv"
        with manifest.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 3200)
        self.assertEqual(sum(row["action"] == "rescore_causal" for row in rows), 640)
        self.assertEqual(sum(row["action"] == "copy_non_causal" for row in rows), 2560)
        self.assertEqual({row["score_version"] for row in rows}, {"causal_content_v1"})

    def test_llama_calibration_reproduces_table_1(self):
        spec = importlib.util.spec_from_file_location("llama_calibration_table", LLAMA_TABLE_SCRIPT)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        rows = module.load_calibration_summary()
        self.assertEqual([row["grad_accum"] for row in rows], [1, 2, 4, 8])
        expected = {
            1: (0.904, 0.853, 0.700, 0.876),
            2: (0.896, 0.867, 0.547, 0.971),
            4: (0.648, 0.587, 0.500, 0.992),
            8: (0.408, 0.240, 0.393, 0.997),
        }
        for row in rows:
            self.assertEqual(row["number_of_episodes"], 25)
            actual = tuple(float(row[metric]) for metric in module.METRICS)
            self.assertEqual(actual, expected[row["grad_accum"]])

    def test_gemma_calibration_reproduces_table_12(self):
        calibration = ARTIFACTS / "gemma/calibration"
        required = {
            "calibration_runs.csv",
            "calibration_by_objective.csv",
            "calibration_summary.csv",
            "selected_schedule.json",
        }
        self.assertTrue(required.issubset({path.name for path in calibration.iterdir()}))
        with (calibration / "calibration_summary.csv").open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 4)
        expected = {
            1: (0.560, 0.527, 0.460, 0.998, 0.636),
            2: (0.316, 0.360, 0.400, 0.998, 0.519),
            4: (0.168, 0.207, 0.220, 0.998, 0.398),
            8: (0.068, 0.180, 0.213, 0.998, 0.365),
        }
        for row in rows:
            grad_accum = int(row["grad_accum"])
            actual = tuple(
                round(float(row[column]), 3)
                for column in (
                    "acquisition",
                    "transfer",
                    "boundedness",
                    "preservation",
                    "balanced_utility",
                )
            )
            self.assertEqual(actual, expected[grad_accum])

    def test_historical_calibration_artifacts_are_excluded(self):
        calibration_dir = ARTIFACTS / "figures/appendix/localization/calibrations"
        self.assertFalse(calibration_dir.exists())
        self.assertFalse(
            (ARTIFACTS / "figures/appendix/localization/gemma_causal_mapping_budget_sensitivity.pdf").exists()
        )
        self.assertFalse((ARTIFACTS / "localization/gemma_budget_sensitivity").exists())

    def test_figure_sources_exist(self):
        for relative in (
            "figures/data/llama",
            "figures/data/gemma",
            "figures/data/localization",
            "figures/data/synthesis",
        ):
            files = [path for path in (ARTIFACTS / relative).rglob("*") if path.is_file()]
            self.assertTrue(files, relative)

    def test_artifact_text_has_no_personal_absolute_paths(self):
        findings = []
        for path in ARTIFACTS.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            if any(pattern.search(text) for pattern in ABSOLUTE_PATH_PATTERNS):
                findings.append(path.relative_to(PROJECT_ROOT).as_posix())
        self.assertEqual(findings, [])

    def test_model_metadata_marks_omitted_features(self):
        metadata_files = list(ARTIFACTS.rglob("model_metadata.json"))
        self.assertEqual(len(metadata_files), 10)
        for path in metadata_files:
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload.get("feature_source_path"), "external/not-shipped", path)

    def test_retained_analysis_code_does_not_depend_on_results(self):
        excluded_legacy = {
            PROJECT_ROOT / "src/train_localized_lora.py",
            PROJECT_ROOT / "scripts/run_localization_pilot.py",
            PROJECT_ROOT / "scripts/patch_train_fullstack_lora_seed.py",
            PROJECT_ROOT / "scripts/rescore_behavioral_policy_jsonl.py",
            PROJECT_ROOT / "scripts/compiler/plot_exp1_geometry_headroom.py",
            PROJECT_ROOT / "scripts/compiler/plot_exp1_multiseed_robustness.py",
            PROJECT_ROOT / "scripts/compiler/plot_exp2_prediction.py",
        }
        legacy_root = "results" + "/"
        findings = []
        for root in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts", PROJECT_ROOT / "tests"):
            for path in root.rglob("*"):
                if not path.is_file() or path.suffix.lower() not in {".py", ".sh"}:
                    continue
                if path in excluded_legacy:
                    continue
                if legacy_root in path.read_text(encoding="utf-8", errors="replace"):
                    findings.append(path.relative_to(PROJECT_ROOT).as_posix())
        self.assertEqual(findings, [])

    def test_manifest_covers_and_hashes_curated_artifacts(self):
        self.assertTrue(MANIFEST.is_file())
        with MANIFEST.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        expected_fields = {
            "release_path",
            "role",
            "experiment",
            "size_bytes",
            "sha256",
            "source_stage",
            "scorer_version",
            "regeneration_command",
        }
        self.assertEqual(set(rows[0]), expected_fields)
        expected_paths = {
            path.relative_to(PROJECT_ROOT).as_posix()
            for path in ARTIFACTS.rglob("*")
            if path.is_file() and path not in {ARTIFACTS / "README.md", MANIFEST}
        }
        manifest_paths = {row["release_path"] for row in rows}
        self.assertEqual(manifest_paths, expected_paths)
        self.assertEqual(len(rows), len(manifest_paths))
        for row in rows:
            path = PROJECT_ROOT / Path(row["release_path"])
            self.assertEqual(int(row["size_bytes"]), path.stat().st_size, row["release_path"])
            self.assertEqual(row["sha256"], sha256(path), row["release_path"])


if __name__ == "__main__":
    unittest.main()
