import importlib.util
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = PROJECT_ROOT / "scripts" / "experiments" / "localization" / "run_cross_model_localization.py"
ANALYSIS_PATH = PROJECT_ROOT / "scripts" / "reproduce" / "analyze_cross_model_geometry.py"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class CrossModelRunnerPlanTest(unittest.TestCase):
    def test_cross_model_specs_use_normalized_layer_windows(self):
        module = load_module("run_cross_model_localization", RUNNER_PATH)
        args = Namespace(
            models=["meta-llama/Llama-3.1-8B-Instruct", "google/gemma-2-9b-it"],
            model_layers=[],
            objectives=["lexical_binding"],
            seeds=[11],
            conditions=["full", "early", "middle", "late"],
            base_rank=8,
            output_root=Path("outputs/cross_model_localization"),
            python_exe="python",
            train_script="src/train_fullstack_lora.py",
            summarize_script="scripts/experiments/localization/summarize_localization_results.py",
            target_modules=["q_proj", "k_proj"],
            num_train_epochs=3,
            learning_rate="2e-4",
            lora_alpha=32,
            lora_dropout=0.05,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            max_length=512,
            max_new_tokens=32,
            torch_dtype="bfloat16",
            device_map="auto",
            bf16=False,
            fp16=False,
            gradient_checkpointing=False,
        )

        specs = module.build_specs(args)
        by_model_condition = {(s.model_slug, s.condition): s for s in specs}

        self.assertEqual(len(specs), 8)
        self.assertEqual(module.normalized_region_width("meta-llama/Llama-3.1-8B-Instruct", {}), 8)
        self.assertEqual(module.normalized_region_width("google/gemma-2-9b-it", {}), 11)
        self.assertEqual(by_model_condition[("llama_3_1_8b_instruct", "early")].run_id, "crossmodel::llama_3_1_8b_instruct::lexical_binding::budget10::early::rank8::seed11")
        self.assertIn("--model_name", by_model_condition[("gemma_2_9b_it", "middle")].train_command)
        self.assertIn("--region_width", by_model_condition[("gemma_2_9b_it", "middle")].train_command)
        self.assertIn("11", by_model_condition[("gemma_2_9b_it", "middle")].train_command)

    def test_summary_discovery_recovers_specs_when_manifest_is_partial(self):
        module = load_module("run_cross_model_localization_discovery", RUNNER_PATH)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "cross_model_localization"
            output_dir = (
                root
                / "qwen2_5_14b_instruct"
                / "lexical_binding"
                / "budget_10"
                / "condition_early"
                / "rank_8"
                / "seed_22"
            )
            output_dir.mkdir(parents=True)
            (output_dir / "summary_localization_by_seed.csv").write_text(
                "learning_type,budget,seed,localization_condition,split,example_mode,scoring_type,n,strict_accuracy,concept_accuracy\n"
                "lexical_binding,10,22,early,id_eval,ALL_MODES,ALL_SCORING_TYPES,1,1.0,1.0\n",
                encoding="utf-8",
            )

            specs = module.discover_specs(root)

        self.assertEqual(len(specs), 1)
        self.assertEqual(specs[0].model_name, "Qwen/Qwen2.5-14B-Instruct")
        self.assertEqual(specs[0].objective, "lexical_binding")
        self.assertEqual(specs[0].condition, "early")
        self.assertEqual(specs[0].seed, 22)


class CrossModelGeometryAnalysisTest(unittest.TestCase):
    def test_geometry_profiles_and_sign_agreement(self):
        module = load_module("analyze_cross_model_geometry", ANALYSIS_PATH)
        rows = []
        for model_slug, offset in [("model_a", 0.0), ("model_b", 0.02)]:
            for condition, acquisition, transfer, boundedness in [
                ("full", 0.80 + offset, 0.70 + offset, 0.90),
                ("early", 0.84 + offset, 0.68 + offset, 0.88),
                ("middle", 0.82 + offset, 0.75 + offset, 0.91),
                ("late", 0.79 + offset, 0.73 + offset, 0.89),
            ]:
                rows.append(
                    {
                        "model_name": model_slug,
                        "model_slug": model_slug,
                        "objective": "lexical_binding",
                        "seed": "11",
                        "condition": condition,
                        "acquisition": str(acquisition),
                        "transfer": str(transfer),
                        "boundedness": str(boundedness),
                    }
                )

        profiles = module.compute_profiles(rows)
        self.assertEqual(len(profiles), 2)
        first = profiles[0]
        self.assertAlmostEqual(first["acquisition_early_minus_full"], 0.04)
        self.assertAlmostEqual(first["transfer_middle_minus_full"], 0.05)

        _penalties, _summary, sign_rows = module.mislocation(rows)
        lexical_sign = [r for r in sign_rows if r["objective"] == "lexical_binding"][0]
        self.assertEqual(lexical_sign["n_models"], 2)
        self.assertEqual(lexical_sign["n_positive"], 2)


if __name__ == "__main__":
    unittest.main()
