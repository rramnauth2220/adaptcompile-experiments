import importlib.util
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = (
    PROJECT_ROOT
    / "scripts"
    / "experiments"
    / "localization"
    / "run_parameter_matched_localization_controls.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("run_parameter_matched_localization_controls", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class ParameterMatchedPlanTest(unittest.TestCase):
    def test_rank_controls_match_expected_layer_rank_products(self):
        module = load_module()
        args = Namespace(
            objectives=["lexical_binding"],
            seeds=[11],
            base_rank=8,
            controls=["localized_expanded_rank", "full_reduced_rank"],
            region_width=8,
            n_layers=32,
            output_root=Path("outputs/parameter_matched_localization"),
            model_name="meta-llama/Llama-3.1-8B-Instruct",
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
        by_key = {(s.control, s.condition): s for s in specs}

        self.assertEqual(len(specs), 8)
        self.assertEqual(by_key[("localized_expanded_rank", "full")].rank, 8)
        self.assertEqual(by_key[("localized_expanded_rank", "early")].rank, 32)
        self.assertEqual(by_key[("localized_expanded_rank", "middle")].rank, 32)
        self.assertEqual(by_key[("localized_expanded_rank", "late")].rank, 32)

        self.assertEqual(by_key[("full_reduced_rank", "full")].rank, 2)
        self.assertEqual(by_key[("full_reduced_rank", "early")].rank, 8)
        self.assertEqual(by_key[("full_reduced_rank", "middle")].rank, 8)
        self.assertEqual(by_key[("full_reduced_rank", "late")].rank, 8)

        for spec in specs:
            expected_product = 256 if spec.control == "localized_expanded_rank" else 64
            self.assertEqual(spec.approximate_rank_layer_product, expected_product)
            self.assertIn("parameter_matched_localization", spec.output_dir.parts)
            self.assertIn("--localization_condition", spec.train_command)
            self.assertIn("--lora_r", spec.train_command)

    def test_summary_discovery_reads_completed_subdirectories(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "parameter_matched_localization"
            summary_dir = (
                root
                / "localized_expanded_rank"
                / "causal_mapping"
                / "budget_10"
                / "condition_middle"
                / "rank_32"
                / "seed_22"
            )
            summary_dir.mkdir(parents=True)
            (summary_dir / "summary_localization_by_seed.csv").write_text(
                "learning_type,budget,seed,localization_condition,split,example_mode,scoring_type,n,strict_accuracy,concept_accuracy\n"
                "causal_mapping,10,22,middle,id_eval,ALL_MODES,ALL_SCORING_TYPES,1,1.0,1.0\n"
                "causal_mapping,10,22,middle,paraphrase_eval,ALL_MODES,ALL_SCORING_TYPES,1,0.8,0.8\n"
                "causal_mapping,10,22,middle,generalization,ALL_MODES,ALL_SCORING_TYPES,1,0.6,0.6\n"
                "causal_mapping,10,22,middle,negative_control,ALL_MODES,ALL_SCORING_TYPES,1,0.4,0.7\n",
                encoding="utf-8",
            )

            specs = module.discover_completed_specs(
                output_root=root,
                controls=["localized_expanded_rank", "full_reduced_rank"],
                n_layers=32,
                region_width=8,
            )
            module.summarize_completed_runs(specs, root)

            by_seed = (root / "summary_parameter_matched_by_seed.csv").read_text(encoding="utf-8")

        self.assertEqual(len(specs), 1)
        self.assertEqual(specs[0].objective, "causal_mapping")
        self.assertEqual(specs[0].condition, "middle")
        self.assertEqual(specs[0].rank, 32)
        self.assertIn("0.9", by_seed)
        self.assertIn("localized_expanded_rank", by_seed)


if __name__ == "__main__":
    unittest.main()
