import argparse
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str, rel_path: str):
    path = PROJECT_ROOT / rel_path
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class GemmaBackboneSpecTest(unittest.TestCase):
    def test_gemma_depth_mapping_and_capacity_matched_config(self):
        from src.compiler_backbones import build_backbone_config_library, resolve_backbone_spec

        spec = resolve_backbone_spec("google/gemma-2-9b-it")
        self.assertEqual(spec.n_layers, 42)
        self.assertEqual(spec.region_width, 11)

        rows = build_backbone_config_library(spec)
        by_id = {row["config_id"]: row for row in rows}

        self.assertEqual(by_id["early__all__r16"]["resolved_layer_indices"], list(range(0, 11)))
        self.assertEqual(by_id["middle__all__r16"]["resolved_layer_indices"], list(range(15, 26)))
        self.assertEqual(by_id["late__all__r16"]["resolved_layer_indices"], list(range(31, 42)))
        self.assertEqual(by_id["full__all__r4"]["lora_r"], 4)
        self.assertEqual(by_id["full__all__r4"]["lora_alpha"], 8)
        self.assertEqual(by_id["full__all__r4"]["n_layers_adapted"], 42)
        self.assertEqual(by_id["full__all__r4"]["approximate_parameter_cost"], 1176)

    def test_semantic_lora_target_discovery(self):
        from src.compiler_backbones import discover_semantic_target_modules

        module_names = [
            "model.layers.0.self_attn.q_proj",
            "model.layers.0.self_attn.k_proj",
            "model.layers.0.self_attn.v_proj",
            "model.layers.0.self_attn.o_proj",
            "model.layers.0.mlp.gate_proj",
            "model.layers.0.mlp.up_proj",
            "model.layers.0.mlp.down_proj",
        ]
        report = discover_semantic_target_modules(module_names)
        self.assertTrue(report["all_present"])
        self.assertEqual(report["missing"], [])


class GemmaReplicationRunnerTest(unittest.TestCase):
    def test_full_phase_is_separate_from_llama_and_counts_expected_runs(self):
        module = load_script("run_compiler_gemma_replication_test", "scripts/experiments/compiler/run_compiler_gemma_replication.py")
        args = argparse.Namespace(
            phase="full",
            python_exe="python",
            model_name_or_path="google/gemma-2-9b-it",
            n_layers=42,
            episode_manifest="data/compiler/episode_manifest.jsonl",
            examples_path="data/prompt_examples.jsonl",
            config_library="data/compiler_gemma/config_library.jsonl",
            results_root="outputs/compiler_gemma",
            outputs_root="outputs/compiler_gemma",
            selected_schedule="missing_selected_schedule.json",
            preservation_examples_path="data/compiler/preservation_examples.jsonl",
            preservation_baseline_cache="outputs/compiler_gemma/preservation/gemma_2_9b_it_baseline.jsonl",
            primary_config_ids=[
                "early__all__r16",
                "middle__all__r16",
                "late__all__r16",
                "full__all__r4",
            ],
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            layers_pattern="layers",
            num_train_epochs=3.0,
            smoke_num_train_epochs=0.25,
            learning_rate="2e-4",
            per_device_train_batch_size=1,
            gradient_accumulation_steps=2,
            allow_uncalibrated_schedule=True,
            force_gradient_accumulation_steps=None,
            grad_accum_candidates=[1, 2, 4, 8],
            max_length=512,
            max_new_tokens=32,
            preservation_max_new_tokens=16,
            preservation_empty_cuda_cache_every=1,
            preservation_disable_generation_cache=True,
            retry_failed_preservation_once=True,
            preservation_retry_cuda_launch_blocking=True,
            preservation_retry_allocator_conf="expandable_segments:True",
            preservation_retry_sleep_seconds=5.0,
            reuse_existing_eval=True,
            torch_dtype="bfloat16",
            device_map="auto",
            gradient_checkpointing=False,
            feature_backend="frozen_model",
            feature_set="full",
            smoke_learning_types=["lexical_binding"],
            smoke_episodes_per_type=1,
            runner_dry_run=True,
        )

        commands = module.build_phase_commands(args)
        joined = "\n".join(module.shell_join(cmd) for cmd in commands)

        self.assertEqual(module.command_count_for_phase("full"), 3200)
        self.assertEqual(len(commands), 5)
        self.assertIn("google/gemma-2-9b-it", joined)
        self.assertIn("outputs/compiler_gemma/full/runs/train", joined.replace("\\", "/"))
        self.assertIn("--region_width 11", joined)
        self.assertIn("--preservation_empty_cuda_cache_every 1", joined)
        self.assertIn("--preservation_disable_generation_cache", joined)
        self.assertIn("--retry_failed_preservation_once", joined)
        self.assertIn("--reuse_existing_eval", joined)
        self.assertNotIn("outputs/compiler/full", joined.replace("\\", "/"))
        self.assertIn("--dry_run", joined)

    def test_inconclusive_schedule_blocks_unless_explicitly_overridden(self):
        module = load_script("run_compiler_gemma_replication_schedule_test", "scripts/experiments/compiler/run_compiler_gemma_replication.py")
        with tempfile.TemporaryDirectory() as tmp:
            schedule = Path(tmp) / "selected_schedule.json"
            schedule.write_text(
                json.dumps(
                    {
                        "status": "inconclusive",
                        "reason": "fixture gate failure",
                        "candidate_summaries": [
                            {
                                "grad_accum": 1,
                                "acquisition": 0.56,
                                "transfer": 0.53,
                                "boundedness": 0.46,
                                "preservation": 0.99,
                                "balanced_utility": 0.64,
                                "failures": 0,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            args = argparse.Namespace(
                phase="pilot",
                python_exe="python",
                model_name_or_path="google/gemma-2-9b-it",
                n_layers=42,
                episode_manifest="data/compiler/episode_manifest.jsonl",
                examples_path="data/prompt_examples.jsonl",
                config_library="data/compiler_gemma/config_library.jsonl",
                results_root="outputs/compiler_gemma",
                outputs_root="outputs/compiler_gemma",
                selected_schedule=str(schedule),
                preservation_examples_path="data/compiler/preservation_examples.jsonl",
                preservation_baseline_cache="outputs/compiler_gemma/preservation/gemma_2_9b_it_baseline.jsonl",
                primary_config_ids=["early__all__r16", "middle__all__r16", "late__all__r16", "full__all__r4"],
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                layers_pattern="layers",
                num_train_epochs=3.0,
                smoke_num_train_epochs=0.25,
                learning_rate="2e-4",
                per_device_train_batch_size=1,
                gradient_accumulation_steps=2,
                allow_uncalibrated_schedule=False,
                force_gradient_accumulation_steps=None,
                grad_accum_candidates=[1, 2, 4, 8],
                max_length=512,
                max_new_tokens=32,
                preservation_max_new_tokens=16,
                preservation_empty_cuda_cache_every=1,
                preservation_disable_generation_cache=True,
                retry_failed_preservation_once=True,
                preservation_retry_cuda_launch_blocking=True,
                preservation_retry_allocator_conf="expandable_segments:True",
                preservation_retry_sleep_seconds=5.0,
                reuse_existing_eval=True,
                torch_dtype="bfloat16",
                device_map="auto",
                gradient_checkpointing=False,
                feature_backend="frozen_model",
                feature_set="full",
                smoke_learning_types=["lexical_binding"],
                smoke_episodes_per_type=1,
                runner_dry_run=True,
            )

            with self.assertRaises(module.ScheduleSelectionError):
                module.build_phase_commands(args)

            args.force_gradient_accumulation_steps = 1
            commands = module.build_phase_commands(args)

        joined = "\n".join(module.shell_join(cmd) for cmd in commands)
        self.assertIn("--gradient_accumulation_steps 1", joined)


class GemmaPilotSummaryTest(unittest.TestCase):
    def test_pilot_summary_uses_tie_aware_oracle_outputs(self):
        module = load_script("summarize_gemma_compiler_replication_test", "scripts/experiments/compiler/summarize_gemma_compiler_replication.py")
        from src.compiler_common import write_jsonl

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            configs = ["early__all__r16", "middle__all__r16", "late__all__r16", "full__all__r4"]
            for idx, config_id in enumerate(configs):
                job_dir = root / "runs" / f"episode_a" / config_id / "seed_11"
                job_dir.mkdir(parents=True)
                metadata = {
                    "episode_id": "compiler::lexical_binding::lexical_0000",
                    "spec_ids": ["lexical_0000"],
                    "meta_split": "train",
                    "learning_type": "lexical_binding",
                    "model_name": "google/gemma-2-9b-it",
                    "model_slug": "gemma_2_9b_it",
                    "config_id": config_id,
                    "seed": 11,
                    "approximate_parameter_cost": 100 + idx,
                }
                (job_dir / "compiler_job_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
                score = 1.0 if config_id in {"early__all__r16", "middle__all__r16"} else 0.25
                write_jsonl(
                    job_dir / "evaluation.jsonl",
                    [
                        {"split": "id_eval", "strict_accuracy": score},
                        {"split": "paraphrase_eval", "strict_accuracy": score},
                        {"split": "generalization", "strict_accuracy": score},
                        {"split": "negative_control", "strict_accuracy": score},
                    ],
                )
                (job_dir / "preservation_summary.json").write_text(
                    json.dumps(
                        {
                            "episode_id": metadata["episode_id"],
                            "config_id": config_id,
                            "seed": 11,
                            "model_name": "google/gemma-2-9b-it",
                            "preservation": score,
                            "n_preservation_total": 1,
                            "n_preservation_baseline_correct": 1,
                            "n_preservation_evaluated": 1,
                            "n_preservation_retained": int(score == 1.0),
                        }
                    ),
                    encoding="utf-8",
                )

            args = argparse.Namespace(
                input_glob=str(root / "runs" / "**" / "evaluation.jsonl"),
                output_dir=str(root / "pilot"),
                config_ids=configs,
                tie_tolerance=1e-12,
            )
            module.command_pilot(args)

            summary = json.loads((root / "pilot" / "selection_summary.json").read_text(encoding="utf-8"))
            winners = (root / "pilot" / "oracle_winners.csv").read_text(encoding="utf-8")

        self.assertAlmostEqual(summary["effectively_tied_episode_fraction"], 1.0)
        self.assertIn("early__all__r16", winners)
        self.assertIn("middle__all__r16", winners)


if __name__ == "__main__":
    unittest.main()
