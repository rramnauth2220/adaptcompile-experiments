import argparse
import csv
import importlib.util
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

try:
    import torch
    from torch import nn
except ModuleNotFoundError:  # pragma: no cover - depends on local environment
    torch = None
    nn = None
TORCH_MODULE_BASE = nn.Module if nn is not None else object

from src.common import write_jsonl
from src.compiler_common import write_csv
import src.extract_episode_features as feature_module
from src.make_compiler_config_library import build_config_library, validate_config_library
from src.make_compiler_episode_manifest import (
    build_episode_manifest,
    validate_episode_manifest,
)
from src.extract_episode_features import (
    chunks,
    discover_probe_modules,
    extract_episode_features,
    finalize_probe_records,
    gradient_sketch_seed,
    make_activation_hook,
    mean_pairwise_cosine,
    parse_layer_and_module,
    resolve_probe_module_batch_size,
    theoretical_probe_passes,
    update_probe_record,
)
from src.evaluate_preservation import (
    build_cache_rows,
    build_preservation_rows,
    render_generation_prompt,
    response_for_scoring,
    strict_correct,
    summarize_preservation_rows,
    validate_baseline_cache_rows,
    validate_preservation_pool,
)
from src.compiler_feature_encoding import encode_episode_config
from src.build_geometry_records import build_records
from src.train_geometry_predictor import (
    apply_standardizer,
    config_mean_baseline,
    fit_standardizer,
)
from src.evaluate_compiler import evaluate_predictions, parse_utility_weights
from src.prepare_geometry_dataset import PRIMARY_CONFIGS, prepare_geometry_dataset
from src.experiment2_geometry_predictor import (
    assert_complete_config_coverage,
    assert_no_split_leakage,
    rank_episode,
    read_feature_payloads,
    run_experiment2,
    sklearn_random_forest_available,
)
from src.experiment2_lofo_generalization import run_lofo


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_script(name, relative_path):
    path = PROJECT_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def toy_examples(objectives=("lexical_binding", "causal_mapping"), n_specs=3, n_train=3):
    rows = []
    split_counts = {
        "train": n_train,
        "id_eval": 1,
        "paraphrase_eval": 1,
        "generalization": 1,
        "negative_control": 1,
    }
    for objective in objectives:
        prefix = objective.split("_")[0]
        for i in range(n_specs):
            spec_id = f"{prefix}_{i:04d}"
            for split, count in split_counts.items():
                for j in range(count):
                    rows.append(
                        {
                            "example_id": f"{spec_id}_{split}_{j}",
                            "spec_id": spec_id,
                            "learning_type": objective,
                            "split": split,
                            "prompt": f"{objective} {spec_id} {split} prompt {j}",
                            "target": f"{objective} target {i}",
                            "scoring": {"scoring_type": split, "exact_answer": f"target {i}"},
                            "metadata": {"train_order": j if split == "train" else None},
                        }
                    )
    return rows


class CompilerEpisodeManifestTest(unittest.TestCase):
    def test_manifest_is_deterministic_stratified_and_leak_free(self):
        examples = toy_examples()
        budgets = {"lexical_binding": 2, "causal_mapping": 2}
        first = build_episode_manifest(
            examples,
            train_count=1,
            validation_count=1,
            test_count=1,
            split_seed=7,
            objectives=["lexical_binding", "causal_mapping"],
            budgets=budgets,
        )
        second = build_episode_manifest(
            examples,
            train_count=1,
            validation_count=1,
            test_count=1,
            split_seed=7,
            objectives=["lexical_binding", "causal_mapping"],
            budgets=budgets,
        )

        self.assertEqual(first, second)
        self.assertEqual(len(first), 6)
        validate_episode_manifest(first, examples, train_count=1, validation_count=1, test_count=1)

        spec_splits = {}
        for row in first:
            self.assertEqual(len(row["spec_ids"]), 1)
            self.assertEqual(len(row["train_example_ids"]), 2)
            spec_id = row["spec_ids"][0]
            self.assertNotIn(spec_id, spec_splits)
            spec_splits[spec_id] = row["meta_split"]

    def test_budget_overflow_and_malformed_manifest_are_rejected(self):
        examples = toy_examples(objectives=("lexical_binding",), n_specs=3, n_train=1)
        with self.assertRaises(ValueError):
            build_episode_manifest(
                examples,
                train_count=1,
                validation_count=1,
                test_count=1,
                objectives=["lexical_binding"],
                budgets={"lexical_binding": 2},
            )

        malformed = [
            {
                "episode_id": "compiler::lexical_binding::lexical_0000",
                "meta_split": "train",
                "learning_type": "lexical_binding",
                "spec_ids": ["lexical_0000"],
            }
        ]
        with self.assertRaises(ValueError):
            validate_episode_manifest(malformed, examples)


class CompilerConfigLibraryTest(unittest.TestCase):
    def test_phase_zero_configs_and_layer_windows(self):
        rows = build_config_library(n_layers=12, region_width=3, lora_r=4, lora_alpha=8)
        validate_config_library(rows)
        by_id = {row["config_id"]: row for row in rows}

        self.assertEqual(
            set(by_id),
            {"full__all__r4", "early__all__r4", "middle__all__r4", "late__all__r4", "full__all__r1"},
        )
        self.assertIsNone(by_id["full__all__r4"]["resolved_layer_indices"])
        self.assertEqual(by_id["early__all__r4"]["resolved_layer_indices"], [0, 1, 2])
        self.assertEqual(by_id["middle__all__r4"]["resolved_layer_indices"], [4, 5, 6])
        self.assertEqual(by_id["late__all__r4"]["resolved_layer_indices"], [9, 10, 11])
        self.assertEqual(by_id["full__all__r4"]["approximate_rank_layer_product"], 48)
        self.assertEqual(by_id["early__all__r4"]["approximate_rank_layer_product"], 12)
        self.assertEqual(by_id["early__all__r4"]["approximate_parameter_cost"], 84)
        self.assertEqual(by_id["full__all__r1"]["lora_alpha"], 2)
        self.assertEqual(
            by_id["full__all__r1"]["approximate_parameter_cost"],
            by_id["early__all__r4"]["approximate_parameter_cost"],
        )

    def test_default_config_library_contains_budget_matched_full_rank(self):
        rows = build_config_library()
        validate_config_library(rows)
        by_id = {row["config_id"]: row for row in rows}

        self.assertEqual(
            set(by_id),
            {"full__all__r16", "early__all__r16", "middle__all__r16", "late__all__r16", "full__all__r4"},
        )
        self.assertEqual(by_id["full__all__r4"]["localization_condition"], "full")
        self.assertEqual(by_id["full__all__r4"]["lora_r"], 4)
        self.assertEqual(by_id["full__all__r4"]["lora_alpha"], 8)
        self.assertEqual(by_id["full__all__r4"]["approximate_parameter_cost"], 896)
        self.assertEqual(
            by_id["full__all__r4"]["approximate_parameter_cost"],
            by_id["early__all__r16"]["approximate_parameter_cost"],
        )


class SpecLevelGeometryDiagnosticTest(unittest.TestCase):
    def test_spec_level_geometry_uses_objective_boundedness_rules(self):
        module = load_script("analyze_spec_level_geometry_test", "scripts/reproduce/analyze_spec_level_geometry.py")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "results.jsonl"
            rows = []
            for condition, strict_neg, concept_neg in [("early", 0, 1), ("late", 1, 0)]:
                for split, strict, concept in [
                    ("id_eval", 1, 1),
                    ("paraphrase_eval", 1, 1),
                    ("generalization", 0, 0),
                    ("negative_control", strict_neg, concept_neg),
                ]:
                    rows.append(
                        {
                            "learning_type": "causal_mapping",
                            "spec_id": "causal_0000",
                            "split": split,
                            "localization_condition": condition,
                            "seed": 11,
                            "strict_accuracy": strict,
                            "concept_accuracy": concept,
                        }
                    )
            write_jsonl(path, rows)

            geom = module.aggregate_geometry([path])
            best = module.best_conditions(geom)
            summary = module.variation_summary(best)

        by_condition = {row["localization_condition"]: row for row in geom}
        self.assertEqual(by_condition["early"]["boundedness"], 1.0)
        self.assertEqual(by_condition["late"]["boundedness"], 0.0)
        bounded_best = [row for row in best if row["metric"] == "boundedness"][0]
        self.assertEqual(bounded_best["best_condition"], "early")
        self.assertTrue(any(row["metric"] == "exploratory_balanced_utility" for row in summary))


class GeometryRecordBuilderTest(unittest.TestCase):
    def test_records_compute_acquisition_boundedness_and_missing_preservation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = root / "evaluation.jsonl"
            metadata = root / "compiler_job_metadata.json"
            metadata.write_text(
                json.dumps(
                    {
                        "episode_id": "compiler::behavioral_policy::behavioral_0000",
                        "config_id": "early__all__r16",
                        "meta_split": "test",
                        "learning_type": "behavioral_policy",
                        "model_name": "toy-model",
                        "model_slug": "toy_model",
                        "seed": 11,
                        "spec_ids": ["behavioral_0000"],
                        "approximate_parameter_cost": 448,
                    }
                ),
                encoding="utf-8",
            )
            rows = []
            for split, strict, concept in [
                ("id_eval", 1, 1),
                ("paraphrase_eval", 0, 0),
                ("generalization", 1, 1),
                ("negative_control", 0, 1),
            ]:
                rows.append(
                    {
                        "learning_type": "behavioral_policy",
                        "spec_id": "behavioral_0000",
                        "split": split,
                        "strict_accuracy": strict,
                        "concept_accuracy": concept,
                    }
                )
            write_jsonl(result, rows)

            records = build_records([str(result)])

        self.assertEqual(len(records), 1)
        rec = records[0]
        self.assertEqual(rec["acquisition"], 0.5)
        self.assertEqual(rec["transfer"], 1.0)
        self.assertEqual(rec["boundedness"], 1.0)
        self.assertIsNone(rec["preservation"])
        self.assertEqual(rec["parameter_cost"], 448)

    def test_records_join_sibling_preservation_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = root / "evaluation.jsonl"
            preservation = root / "preservation.jsonl"
            metadata = root / "compiler_job_metadata.json"
            metadata.write_text(
                json.dumps(
                    {
                        "episode_id": "compiler::lexical_binding::lexical_0000",
                        "config_id": "early__all__r16",
                        "meta_split": "test",
                        "learning_type": "lexical_binding",
                        "model_name": "toy-model",
                        "model_slug": "toy_model",
                        "seed": 11,
                        "spec_ids": ["lexical_0000"],
                        "approximate_parameter_cost": 448,
                    }
                ),
                encoding="utf-8",
            )
            rows = []
            for split, strict in [
                ("id_eval", 1),
                ("paraphrase_eval", 1),
                ("generalization", 1),
                ("negative_control", 1),
            ]:
                rows.append(
                    {
                        "learning_type": "lexical_binding",
                        "spec_id": "lexical_0000",
                        "split": split,
                        "strict_accuracy": strict,
                    }
                )
            write_jsonl(result, rows)
            write_jsonl(
                preservation,
                [
                    {
                        "episode_id": "compiler::lexical_binding::lexical_0000",
                        "config_id": "early__all__r16",
                        "seed": 11,
                        "example_id": "preserve_1",
                        "preservation_correct": 1,
                        "preservation": 0.5,
                        "n_preservation_total": 3,
                        "n_preservation_baseline_correct": 2,
                        "n_preservation_evaluated": 2,
                    },
                    {
                        "episode_id": "compiler::lexical_binding::lexical_0000",
                        "config_id": "early__all__r16",
                        "seed": 11,
                        "example_id": "preserve_2",
                        "preservation_correct": 0,
                        "preservation": 0.5,
                        "n_preservation_total": 3,
                        "n_preservation_baseline_correct": 2,
                        "n_preservation_evaluated": 2,
                    },
                ],
            )

            records = build_records([str(result)])

        self.assertEqual(len(records), 1)
        rec = records[0]
        self.assertEqual(rec["preservation"], 0.5)
        self.assertEqual(rec["n_preservation_total"], 3)
        self.assertEqual(rec["n_preservation_baseline_correct"], 2)
        self.assertEqual(rec["n_preservation_evaluated"], 2)
        self.assertEqual(rec["n_preservation_retained"], 1)
        self.assertEqual(rec["exploratory_balanced_utility"], 0.875)

    def test_records_prefer_sibling_preservation_summary_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = root / "evaluation.jsonl"
            summary = root / "preservation_summary.json"
            metadata = root / "compiler_job_metadata.json"
            metadata.write_text(
                json.dumps(
                    {
                        "episode_id": "compiler::lexical_binding::lexical_0000",
                        "config_id": "early__all__r16",
                        "meta_split": "test",
                        "learning_type": "lexical_binding",
                        "model_name": "toy-model",
                        "model_slug": "toy_model",
                        "seed": 11,
                        "spec_ids": ["lexical_0000"],
                    }
                ),
                encoding="utf-8",
            )
            write_jsonl(
                result,
                [
                    {
                        "split": split,
                        "strict_accuracy": 1,
                        "learning_type": "lexical_binding",
                        "spec_id": "lexical_0000",
                    }
                    for split in ["id_eval", "paraphrase_eval", "generalization", "negative_control"]
                ],
            )
            summary.write_text(
                json.dumps(
                    {
                        "episode_id": "compiler::lexical_binding::lexical_0000",
                        "config_id": "early__all__r16",
                        "seed": 11,
                        "model_name": "toy-model",
                        "preservation": 2 / 3,
                        "n_preservation_total": 5,
                        "n_preservation_baseline_correct": 3,
                        "n_preservation_evaluated": 3,
                        "n_preservation_retained": 2,
                        "category_preservation": {"arithmetic_and_comparison": 0.5},
                    }
                ),
                encoding="utf-8",
            )

            records = build_records([str(result)])

        rec = records[0]
        self.assertAlmostEqual(rec["preservation"], 2 / 3)
        self.assertEqual(rec["n_preservation_total"], 5)
        self.assertEqual(rec["n_preservation_baseline_correct"], 3)
        self.assertEqual(rec["n_preservation_evaluated"], 3)
        self.assertEqual(rec["n_preservation_retained"], 2)
        self.assertAlmostEqual(rec["exploratory_balanced_utility"], (1 + 1 + 1 + 2 / 3) / 4)


class PreservationEvaluatorTest(unittest.TestCase):
    def test_frozen_correct_filtering_and_preservation_calculation(self):
        examples = [
            {
                "example_id": "p1",
                "category": "arithmetic_and_comparison",
                "scorer": "normalized_exact_match",
                "prompt": "Name the retained label.",
                "target": "alpha",
            },
            {
                "example_id": "p2",
                "category": "arithmetic_and_comparison",
                "scorer": "normalized_exact_match",
                "prompt": "Name the other retained label.",
                "target": "beta",
            },
            {
                "example_id": "p3",
                "category": "logical_inference",
                "scorer": "normalized_exact_match",
                "prompt": "Name the third retained label.",
                "target": "gamma",
            },
            {
                "example_id": "p4",
                "category": "logical_inference",
                "scorer": "normalized_exact_match",
                "prompt": "Name the first frozen-wrong label.",
                "target": "delta",
            },
            {
                "example_id": "p5",
                "category": "stable_common_knowledge",
                "scorer": "normalized_exact_match",
                "prompt": "Name the second frozen-wrong label.",
                "target": "epsilon",
            },
        ]

        cache_rows = build_cache_rows(
            examples,
            {"p1": "alpha", "p2": "beta.", "p3": "gamma", "p4": "wrong", "p5": "epsilon plus"},
            model_name="toy-model",
        )
        self.assertEqual(sum(1 for row in cache_rows if row["frozen_correct"]), 3)

        rows = build_preservation_rows(
            cache_rows,
            {"p1": "alpha", "p2": "wrong", "p3": "gamma"},
            episode_id="compiler::lexical_binding::lexical_0000",
            config_id="early__all__r16",
            seed=11,
            model_name="toy-model",
        )
        summary = summarize_preservation_rows(
            rows,
            episode_id="compiler::lexical_binding::lexical_0000",
            config_id="early__all__r16",
            seed=11,
            model_name="toy-model",
        )

        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0]["n_preservation_total"], 5)
        self.assertEqual(rows[0]["n_preservation_baseline_correct"], 3)
        self.assertEqual(rows[0]["n_preservation_evaluated"], 3)
        self.assertEqual(rows[0]["n_preservation_retained"], 2)
        self.assertAlmostEqual(rows[0]["preservation"], 2 / 3)
        self.assertAlmostEqual(summary["preservation"], 2 / 3)
        self.assertEqual(summary["n_preservation_retained"], 2)
        self.assertAlmostEqual(summary["category_preservation"]["arithmetic_and_comparison"], 0.5)
        self.assertAlmostEqual(summary["category_preservation"]["logical_inference"], 1.0)

    def test_preservation_exact_match_is_shared_for_frozen_and_adapted_scoring(self):
        example = {
            "example_id": "p1",
            "prompt": "Return the label.",
            "target": "Alpha",
            "scorer": "normalized_exact_match",
        }

        self.assertTrue(strict_correct(example, " alpha. ")[0])
        self.assertFalse(strict_correct(example, "not alpha")[0])
        cache_rows = build_cache_rows([example], {"p1": " alpha. "}, model_name="toy-model")
        preserved = build_preservation_rows(
            cache_rows,
            {"p1": "not alpha"},
            episode_id="compiler::lexical_binding::lexical_0000",
            config_id="early__all__r16",
            seed=11,
        )
        self.assertTrue(cache_rows[0]["frozen_correct"])
        self.assertFalse(preserved[0]["adapted_correct"])
        self.assertEqual(preserved[0]["preservation"], 0.0)

    def test_first_line_response_extraction_stays_exact(self):
        example = {
            "example_id": "p1",
            "prompt": "Return the label.",
            "target": "alpha",
            "scorer": "normalized_exact_match",
        }

        self.assertEqual(response_for_scoring("alpha\nbecause...", "first_line"), "alpha")
        self.assertTrue(strict_correct(example, "alpha\nbecause...", response_extraction="first_line")[0])
        self.assertFalse(strict_correct(example, "alpha\nbecause...", response_extraction="full")[0])
        self.assertFalse(strict_correct(example, "the answer is alpha", response_extraction="first_line")[0])

    def test_auto_prompt_format_uses_chat_template_when_available(self):
        class ToyTokenizer:
            chat_template = "toy"

            def apply_chat_template(self, messages, tokenize, add_generation_prompt):
                self.messages = messages
                self.tokenize = tokenize
                self.add_generation_prompt = add_generation_prompt
                return f"<chat>{messages[0]['content']}</chat>"

        tokenizer = ToyTokenizer()
        example = {"example_id": "p1", "prompt": "Say alpha.", "target": "alpha"}

        self.assertEqual(render_generation_prompt(example, tokenizer, prompt_format="auto"), "<chat>Say alpha.</chat>")
        self.assertEqual(tokenizer.messages, [{"role": "user", "content": "Say alpha."}])
        self.assertEqual(render_generation_prompt(example, object(), prompt_format="auto"), "### Prompt:\nSay alpha.\n\n### Answer:\n")

    def test_preservation_pool_validation_rejects_empty_and_leaked_specs(self):
        with self.assertRaises(ValueError):
            validate_preservation_pool([])
        with self.assertRaises(ValueError):
            validate_preservation_pool(
                [{"example_id": "p1", "spec_id": "target_spec", "prompt": "p", "target": "t"}],
                forbidden_spec_ids=["target_spec"],
            )

    def test_cache_errors_when_no_frozen_correct_examples(self):
        examples = [{"example_id": "p1", "prompt": "Name the label.", "target": "alpha"}]
        with self.assertRaises(ValueError):
            build_cache_rows(examples, {"p1": "wrong"}, model_name="toy-model")

    def test_duplicate_preservation_ids_error(self):
        examples = [
            {"example_id": "p1", "prompt": "Prompt A", "target": "alpha"},
            {"example_id": "p1", "prompt": "Prompt B", "target": "beta"},
        ]
        with self.assertRaisesRegex(ValueError, "Duplicate preservation example_id"):
            validate_preservation_pool(examples)

    def test_baseline_cache_validation_rejects_id_and_model_mismatches(self):
        examples = [
            {"example_id": "p1", "prompt": "Prompt A", "target": "alpha"},
            {"example_id": "p2", "prompt": "Prompt B", "target": "beta"},
        ]
        cache_rows = build_cache_rows(examples, {"p1": "alpha", "p2": "beta"}, model_name="toy-model")

        missing = cache_rows[:1]
        with self.assertRaisesRegex(ValueError, "do not match"):
            validate_baseline_cache_rows(missing, examples, model_name="toy-model")

        extra = cache_rows + [
            {
                **cache_rows[0],
                "example_id": "p3",
                "prompt": "Prompt C",
                "target": "gamma",
            }
        ]
        with self.assertRaisesRegex(ValueError, "do not match"):
            validate_baseline_cache_rows(extra, examples, model_name="toy-model")

        wrong_model = [dict(row) for row in cache_rows]
        wrong_model[0]["model_name"] = "other-model"
        with self.assertRaisesRegex(ValueError, "model mismatch"):
            validate_baseline_cache_rows(wrong_model, examples, model_name="toy-model")


class FeatureEncodingTest(unittest.TestCase):
    def test_feature_schema_excludes_learning_type_and_encodes_config_cost(self):
        examples = toy_examples(objectives=("lexical_binding",), n_specs=3)
        rows = build_episode_manifest(
            examples,
            train_count=1,
            validation_count=1,
            test_count=1,
            objectives=["lexical_binding"],
            budgets={"lexical_binding": 2},
        )
        episode = rows[0]
        examples_by_id = {row["example_id"]: row for row in examples}
        features = extract_episode_features(
            episode,
            examples_by_id=examples_by_id,
            model_name_or_path="toy/model",
            backend="proxy",
            n_layers=8,
            seed=3,
        )
        config = build_config_library(n_layers=8, region_width=2, lora_r=4)[1]
        x, names = encode_episode_config(features, config)

        self.assertIn("learning_type", features["metadata"])
        self.assertNotIn("learning_type", features["episode_features"])
        self.assertFalse(any("learning_type" in name for name in names))
        self.assertEqual(len(x), len(names))
        self.assertIn("config__approximate_parameter_cost", names)
        self.assertGreater(x[names.index("selected__n_modules")], 0)

    def test_train_only_standardizer(self):
        standardizer = fit_standardizer([[1.0, 10.0], [3.0, 14.0]])
        transformed = apply_standardizer([[2.0, 12.0], [100.0, 200.0]], standardizer)

        self.assertEqual(standardizer["mean"], [2.0, 12.0])
        self.assertAlmostEqual(transformed[0][0], 0.0)
        self.assertGreater(transformed[1][0], 10.0)

    def test_config_only_baseline_uses_training_config_means(self):
        train_rows = [
            {"record": {"config_id": "early"}, "y": [0.2, 0.4, 0.6]},
            {"record": {"config_id": "early"}, "y": [0.4, 0.6, 0.8]},
            {"record": {"config_id": "late"}, "y": [0.9, 0.9, 0.9]},
        ]
        eval_rows = [{"record": {"config_id": "early"}, "y": [0, 0, 0]}]
        pred = config_mean_baseline(train_rows, eval_rows)
        self.assertEqual(pred[0], [0.30000000000000004, 0.5, 0.7])

    def test_frozen_backend_helper_parses_modules_and_agreement_sketches(self):
        parsed = parse_layer_and_module(
            "model.layers.12.self_attn.q_proj",
            ["q_proj", "v_proj"],
        )
        self.assertEqual(parsed, (12, "q_proj"))
        self.assertIsNone(parse_layer_and_module("model.layers.12.self_attn.rotary_emb", ["q_proj"]))

        self.assertAlmostEqual(mean_pairwise_cosine([[1.0, 0.0], [1.0, 0.0]]), 1.0)
        self.assertAlmostEqual(mean_pairwise_cosine([[1.0, 0.0], [0.0, 1.0]]), 0.0)

    def test_gradient_sketch_projection_seed_ignores_example_index(self):
        seed_a = gradient_sketch_seed(
            2026,
            "compiler::lexical_binding::lexical_0000",
            "0000::q_proj::model.layers.0.self_attn.q_proj",
            example_idx=0,
        )
        seed_b = gradient_sketch_seed(
            2026,
            "compiler::lexical_binding::lexical_0000",
            "0000::q_proj::model.layers.0.self_attn.q_proj",
            example_idx=1,
        )
        seed_c = gradient_sketch_seed(
            2027,
            "compiler::lexical_binding::lexical_0000",
            "0000::q_proj::model.layers.0.self_attn.q_proj",
            example_idx=1,
        )
        seed_d = gradient_sketch_seed(
            2026,
            "compiler::lexical_binding::lexical_0000",
            "0001::v_proj::model.layers.1.self_attn.v_proj",
            example_idx=1,
        )

        self.assertEqual(seed_a, seed_b)
        self.assertNotEqual(seed_a, seed_c)
        self.assertNotEqual(seed_a, seed_d)

    def test_proxy_feature_extraction_embedding_is_deterministic_and_metadata_only(self):
        examples = toy_examples(objectives=("lexical_binding",), n_specs=3)
        rows = build_episode_manifest(
            examples,
            train_count=1,
            validation_count=1,
            test_count=1,
            objectives=["lexical_binding"],
            budgets={"lexical_binding": 2},
        )
        examples_by_id = {row["example_id"]: row for row in examples}
        first = extract_episode_features(
            rows[0],
            examples_by_id=examples_by_id,
            model_name_or_path="toy/model",
            backend="proxy",
            n_layers=4,
            seed=17,
            episode_embedding_dim=8,
        )
        second = extract_episode_features(
            rows[0],
            examples_by_id=examples_by_id,
            model_name_or_path="toy/model",
            backend="proxy",
            n_layers=4,
            seed=17,
            episode_embedding_dim=8,
        )
        embedding_keys = sorted(k for k in first["episode_features"] if k.startswith("episode_embedding_"))

        self.assertEqual(len(embedding_keys), 8)
        self.assertEqual(
            [first["episode_features"][k] for k in embedding_keys],
            [second["episode_features"][k] for k in embedding_keys],
        )
        self.assertIn("learning_type", first["metadata"])
        self.assertNotIn("learning_type", first["episode_features"])
        self.assertEqual(first["metadata"]["episode_representation_projection_dim"], 8)

    def test_feature_sets_select_expected_feature_groups(self):
        features = {
            "metadata": {"n_layers": 4},
            "episode_features": {
                "n_adaptation_examples": 2,
                "frozen_target_token_loss_mean": 1.3,
                "episode_embedding_000": 0.25,
            },
            "module_features": [
                {
                    "layer_index": 0,
                    "normalized_depth": 0.0,
                    "module_type": "q_proj",
                    "sensitivity_proxy": 0.5,
                    "gradient_magnitude_mean": 0.3,
                    "activation_rms_mean": 0.7,
                    "gradient_agreement_proxy": 0.9,
                },
                {
                    "layer_index": 3,
                    "normalized_depth": 1.0,
                    "module_type": "q_proj",
                    "sensitivity_proxy": 1.5,
                    "gradient_magnitude_mean": 1.3,
                    "activation_rms_mean": 1.7,
                    "gradient_agreement_proxy": 0.4,
                }
            ],
        }
        config = {
            "config_id": "early__all__r16",
            "localization_condition": "early",
            "module_family": "all",
            "target_modules": ["q_proj"],
            "lora_r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "region_width": 1,
            "n_layers": 4,
        }

        _x_episode, episode_names = encode_episode_config(features, config, feature_set="episode")
        _x_frozen, frozen_names = encode_episode_config(features, config, feature_set="frozen_behavior")
        x_probe, probe_names = encode_episode_config(features, config, feature_set="module_probes")
        late_config = {**config, "localization_condition": "late", "resolved_layer_indices": [3]}
        x_late, late_names = encode_episode_config(features, late_config, feature_set="module_probes")

        self.assertIn("episode__n_adaptation_examples", episode_names)
        self.assertIn("episode__episode_embedding_000", episode_names)
        self.assertNotIn("episode__frozen_target_token_loss_mean", episode_names)
        self.assertIn("episode__frozen_target_token_loss_mean", frozen_names)
        self.assertNotIn("episode__n_adaptation_examples", frozen_names)
        self.assertIn("selected__sensitivity_mean", probe_names)
        self.assertIn("selected__gradient_magnitude_mean", probe_names)
        self.assertIn("selected__activation_rms_mean", probe_names)
        self.assertIn("config__is_localized", probe_names)
        self.assertTrue(all(math.isfinite(value) for value in x_probe))
        self.assertEqual(probe_names, late_names)
        self.assertNotEqual(
            x_probe[probe_names.index("selected__depth_mean")],
            x_late[late_names.index("selected__depth_mean")],
        )

    def test_feature_encoding_rejects_zero_selected_modules(self):
        features = {
            "metadata": {"n_layers": 2},
            "episode_features": {"n_adaptation_examples": 1},
            "module_features": [
                {"layer_index": 0, "normalized_depth": 0.0, "module_type": "q_proj"}
            ],
        }
        config = {
            "config_id": "bad__all__r1",
            "localization_condition": "late",
            "resolved_layer_indices": [1],
            "module_family": "all",
            "target_modules": ["v_proj"],
            "lora_r": 1,
            "lora_alpha": 2,
            "region_width": 1,
            "n_layers": 2,
        }

        with self.assertRaisesRegex(ValueError, "selected zero probe modules"):
            encode_episode_config(features, config, feature_set="module_probes")

    def test_cli_reuses_frozen_backend_across_multiple_episodes(self):
        examples = toy_examples(objectives=("lexical_binding",), n_specs=2)
        episodes = [
            {
                "episode_id": "compiler::lexical_binding::a",
                "spec_ids": ["lexical_binding_0000"],
                "meta_split": "train",
                "learning_type": "lexical_binding",
                "train_example_ids": [examples[0]["example_id"]],
            },
            {
                "episode_id": "compiler::lexical_binding::b",
                "spec_ids": ["lexical_binding_0001"],
                "meta_split": "train",
                "learning_type": "lexical_binding",
                "train_example_ids": [examples[1]["example_id"]],
            },
        ]

        class FakeFrozenBackend:
            init_calls = 0
            extract_calls = []

            def __init__(self, **_kwargs):
                FakeFrozenBackend.init_calls += 1

            def extract(self, episode, examples_by_id):
                FakeFrozenBackend.extract_calls.append(episode["episode_id"])
                if episode["train_example_ids"][0] not in examples_by_id:
                    raise AssertionError("episode train example missing from examples_by_id")
                return {
                    "schema_version": 1,
                    "metadata": {"episode_id": episode["episode_id"]},
                    "episode_features": {"n_adaptation_examples": 1},
                    "module_features": [],
                }

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            episode_manifest = tmp_path / "episodes.jsonl"
            examples_path = tmp_path / "examples.jsonl"
            output_root = tmp_path / "features"
            write_jsonl(episode_manifest, episodes)
            write_jsonl(examples_path, examples)

            argv = [
                "extract_episode_features.py",
                "--episode_manifest",
                str(episode_manifest),
                "--examples_path",
                str(examples_path),
                "--model_name_or_path",
                "toy/model",
                "--output_root",
                str(output_root),
                "--backend",
                "frozen_model",
            ]
            with mock.patch.object(sys, "argv", argv), mock.patch.object(
                feature_module,
                "FrozenModelFeatureBackend",
                FakeFrozenBackend,
            ), mock.patch(
                "builtins.print",
            ):
                feature_module.main()

            self.assertEqual(FakeFrozenBackend.init_calls, 1)
            self.assertEqual(FakeFrozenBackend.extract_calls, [row["episode_id"] for row in episodes])
            for row in episodes:
                self.assertTrue(feature_module.feature_path(output_root, "toy/model", row["episode_id"]).exists())


class Experiment2GeometryPredictionTest(unittest.TestCase):
    def feature_payload(self, episode_id, split, offset):
        return {
            "schema_version": 1,
            "metadata": {
                "episode_id": episode_id,
                "meta_split": split,
                "model_name": "toy/model",
                "model_slug": "model",
                "n_layers": 32,
                "learning_type": "metadata_only",
            },
            "episode_features": {
                "n_adaptation_examples": 2,
                "target_token_count_mean": 4.0 + offset,
                "frozen_target_token_loss_mean": 1.0 + offset,
                "frozen_target_token_loss_std": 0.1,
            },
            "module_features": [
                {
                    "layer_index": layer,
                    "normalized_depth": layer / 31.0,
                    "module_type": module,
                    "sensitivity_proxy": 0.2 + offset + 0.01 * layer,
                    "gradient_agreement_proxy": 0.8 - 0.01 * offset,
                }
                for layer in (0, 15, 31)
                for module in ("q_proj", "up_proj")
            ],
        }

    def geometry_rows(self):
        episodes = [
            ("ep_train_a", "train", "lexical_binding", 0.0),
            ("ep_train_b", "train", "causal_mapping", 0.1),
            ("ep_val", "validation", "lexical_binding", 0.2),
            ("ep_test", "test", "causal_mapping", 0.3),
        ]
        config_bonus = {
            "early__all__r16": 0.03,
            "middle__all__r16": 0.06,
            "late__all__r16": 0.09,
            "full__all__r4": 0.12,
        }
        rows = []
        for episode_id, split, learning_type, base in episodes:
            for cfg in PRIMARY_CONFIGS:
                for seed in ([11, 22] if split == "test" else [11]):
                    value = 0.2 + base + config_bonus[cfg] + (0.01 if seed == 22 else 0.0)
                    rows.append(
                        {
                            "episode_id": episode_id,
                            "meta_split": split,
                            "learning_type": learning_type,
                            "model_name": "toy/model",
                            "model_slug": "model",
                            "seed": seed,
                            "config_id": cfg,
                            "acquisition": value,
                            "transfer": value + 0.01,
                            "boundedness": value + 0.02,
                            "preservation": value + 0.03,
                        }
                    )
        return rows

    def lofo_geometry_rows(self):
        rows = []
        objectives = ["lexical_binding", "factual_association", "causal_mapping"]
        config_bonus = {
            "early__all__r16": 0.01,
            "middle__all__r16": 0.03,
            "late__all__r16": 0.05,
            "full__all__r4": 0.07,
        }
        objective_bonus = {
            "lexical_binding": 0.00,
            "factual_association": 0.10,
            "causal_mapping": 0.20,
        }
        for objective in objectives:
            for split, n_episodes in [("train", 2), ("validation", 1), ("test", 1)]:
                for i in range(n_episodes):
                    episode_id = f"{objective}_{split}_{i}"
                    base = 0.2 + objective_bonus[objective] + 0.02 * i
                    if split == "validation":
                        base += 0.03
                    elif split == "test":
                        base += 0.06
                    for cfg in PRIMARY_CONFIGS:
                        value = base + config_bonus[cfg]
                        rows.append(
                            {
                                "model_slug": "model",
                                "episode_id": episode_id,
                                "learning_type": objective,
                                "meta_split": split,
                                "model_name": "toy/model",
                                "config_id": cfg,
                                "acquisition": value,
                                "transfer": value + 0.01,
                                "boundedness": value + 0.02,
                                "preservation": value + 0.03,
                                "utility": value + 0.015,
                                "number_of_seeds": 1,
                                "seed_values": "11",
                            }
                        )
        return rows

    def lofo_features(self):
        rows = []
        for row in self.lofo_geometry_rows():
            episode_id = row["episode_id"]
            if any(payload["metadata"]["episode_id"] == episode_id for payload in rows):
                continue
            offset = len(rows) / 100.0
            rows.append(self.feature_payload(episode_id, row["meta_split"], offset))
        return rows

    def test_prepare_geometry_dataset_seed_aggregates_and_reports_coverage(self):
        rows, diagnostics = prepare_geometry_dataset(
            self.geometry_rows(),
            config_ids=PRIMARY_CONFIGS,
            expected_seeds_by_split={"train": {11}, "validation": {11}, "test": {11, 22}},
        )
        test_rows = [row for row in rows if row["meta_split"] == "test"]

        self.assertEqual(len(test_rows), len(PRIMARY_CONFIGS))
        self.assertTrue(all(row["number_of_seeds"] == 2 for row in test_rows))
        self.assertEqual(diagnostics["n_incomplete_episodes"], 0)
        self.assertEqual(diagnostics["complete_episode_counts_by_split"]["test"], 1)
        early = next(row for row in test_rows if row["config_id"] == "early__all__r16")
        self.assertAlmostEqual(early["acquisition"], (0.2 + 0.3 + 0.03 + 0.2 + 0.3 + 0.03 + 0.01) / 2)
        self.assertAlmostEqual(
            early["utility"],
            (
                early["acquisition"]
                + early["transfer"]
                + early["boundedness"]
                + early["preservation"]
            )
            / 4,
        )

    def test_prepare_geometry_dataset_rejects_duplicate_seed_and_missing_expected_seed(self):
        duplicate_rows = self.geometry_rows()
        duplicate_rows.append(dict(duplicate_rows[0]))
        with self.assertRaisesRegex(ValueError, "Duplicate raw geometry row"):
            prepare_geometry_dataset(
                duplicate_rows,
                config_ids=PRIMARY_CONFIGS,
                expected_seeds_by_split={"train": {11}, "validation": {11}, "test": {11, 22}},
            )

        with self.assertRaisesRegex(ValueError, "Unexpected seed coverage"):
            prepare_geometry_dataset(
                self.geometry_rows(),
                config_ids=PRIMARY_CONFIGS,
                expected_seeds_by_split={"train": {11}, "validation": {11}, "test": {11, 22, 33}},
            )

    def test_prepare_geometry_dataset_rejects_missing_config_in_strict_mode(self):
        rows = [
            row
            for row in self.geometry_rows()
            if not (row["episode_id"] == "ep_test" and row["config_id"] == "full__all__r4")
        ]

        with self.assertRaisesRegex(ValueError, "complete config coverage"):
            prepare_geometry_dataset(
                rows,
                config_ids=PRIMARY_CONFIGS,
                expected_seeds_by_split={"train": {11}, "validation": {11}, "test": {11, 22}},
            )

    def test_leakage_and_config_coverage_checks_fail_loudly(self):
        overlapping = [
            {"episode_id": "same", "meta_split": "train", "config_id": PRIMARY_CONFIGS[0], "number_of_seeds": 1},
            {"episode_id": "same", "meta_split": "test", "config_id": PRIMARY_CONFIGS[0], "number_of_seeds": 1},
        ]
        with self.assertRaisesRegex(ValueError, "overlap"):
            assert_no_split_leakage(overlapping)

        incomplete = [
            {"episode_id": "e", "meta_split": "test", "config_id": PRIMARY_CONFIGS[0], "number_of_seeds": 1}
        ]
        with self.assertRaisesRegex(ValueError, "complete config coverage"):
            assert_complete_config_coverage(incomplete, PRIMARY_CONFIGS, split="test")

    def test_tie_aware_ranking_logic(self):
        rows = [
            {
                "episode_id": "e",
                "learning_type": "lexical_binding",
                "config_id": "a",
                "observed_utility": 0.8,
                "primary_predicted_utility": 0.1,
            },
            {
                "episode_id": "e",
                "learning_type": "lexical_binding",
                "config_id": "b",
                "observed_utility": 0.8,
                "primary_predicted_utility": 0.9,
            },
            {
                "episode_id": "e",
                "learning_type": "lexical_binding",
                "config_id": "c",
                "observed_utility": 0.2,
                "primary_predicted_utility": 0.3,
            },
        ]
        detail = rank_episode(rows, method="primary", tolerance=1e-12)

        self.assertEqual(detail["observed_oracle_set"], "a|b")
        self.assertTrue(detail["top1_set_oracle_recovery"])
        self.assertEqual(detail["selected_config"], "b")
        self.assertTrue(detail["selected_config_oracle_optimal"])

    def test_experiment2_predictor_writes_four_outcomes_without_nans(self):
        rows, _diagnostics = prepare_geometry_dataset(self.geometry_rows(), config_ids=PRIMARY_CONFIGS)
        features = [
            self.feature_payload("ep_train_a", "train", 0.0),
            self.feature_payload("ep_train_b", "train", 0.1),
            self.feature_payload("ep_val", "validation", 0.2),
            self.feature_payload("ep_test", "test", 0.3),
        ]
        configs = build_config_library()

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            geometry_path = tmp_path / "geometry.csv"
            features_path = tmp_path / "features.jsonl"
            config_path = tmp_path / "configs.jsonl"
            output_dir = tmp_path / "out"
            write_csv(
                geometry_path,
                rows,
                [
                    "episode_id",
                    "learning_type",
                    "meta_split",
                    "model_name",
                    "model_slug",
                    "config_id",
                    "acquisition",
                    "transfer",
                    "boundedness",
                    "preservation",
                    "utility",
                    "number_of_seeds",
                    "seed_values",
                ],
            )
            write_jsonl(features_path, features)
            write_jsonl(config_path, configs)

            args = argparse.Namespace(
                geometry=str(geometry_path),
                features=str(features_path),
                config_library=str(config_path),
                feature_set="full",
                model_type="auto",
                output_dir=str(output_dir),
                config_ids=PRIMARY_CONFIGS,
                seed=2026,
                ridge_alphas=[0.0, 0.1],
                rf_n_estimators=[5],
                rf_max_depth=["3"],
                rf_min_samples_leaf=[1],
                allow_train_only_smoke_test=False,
                tie_tolerance=1e-12,
            )
            result = run_experiment2(args)

            self.assertIn("primary", result["prediction_metrics"])
            predictions = read_feature_payloads(features_path)
            self.assertIn("ep_test", predictions)
            with (output_dir / "predictions_test.csv").open(encoding="utf-8-sig", newline="") as f:
                prediction_rows = list(csv.DictReader(f))

            self.assertEqual(len(prediction_rows), len(PRIMARY_CONFIGS))
            for outcome in ("acquisition", "transfer", "boundedness", "preservation"):
                self.assertIn(f"observed_{outcome}", prediction_rows[0])
                self.assertIn(f"predicted_{outcome}", prediction_rows[0])
                self.assertNotEqual(prediction_rows[0][f"predicted_{outcome}"], "")
            self.assertTrue((output_dir / "feature_standardizer.json").exists())
            self.assertTrue((output_dir / "trained_model_ridge.npz").exists())

    def test_lofo_holds_out_objective_and_records_leakage_guards(self):
        rows = self.lofo_geometry_rows()
        features = self.lofo_features()

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            geometry_path = tmp_path / "geometry.csv"
            features_path = tmp_path / "features.jsonl"
            config_path = tmp_path / "configs.jsonl"
            output_dir = tmp_path / "lofo"
            write_csv(
                geometry_path,
                rows,
                [
                    "model_slug",
                    "episode_id",
                    "learning_type",
                    "meta_split",
                    "model_name",
                    "config_id",
                    "acquisition",
                    "transfer",
                    "boundedness",
                    "preservation",
                    "utility",
                    "number_of_seeds",
                    "seed_values",
                ],
            )
            write_jsonl(features_path, features)
            write_jsonl(config_path, build_config_library())

            args = argparse.Namespace(
                geometry=str(geometry_path),
                features=str(features_path),
                config_library=str(config_path),
                feature_set="episode",
                model_type="ridge",
                output_dir=str(output_dir),
                config_ids=PRIMARY_CONFIGS,
                held_out_learning_type="lexical_binding",
                seed=2026,
                ridge_alphas=[0.1],
                rf_n_estimators=[5],
                rf_max_depth=["3"],
                rf_min_samples_leaf=[1],
                tie_tolerance=1e-12,
            )
            run_lofo(args)

            fold_dir = output_dir / "lexical_binding"
            metadata = json.loads((fold_dir / "model_metadata.json").read_text(encoding="utf-8"))
            standardizer = json.loads((fold_dir / "feature_standardizer.json").read_text(encoding="utf-8"))
            with (fold_dir / "predictions_test.csv").open(encoding="utf-8-sig", newline="") as f:
                prediction_rows = list(csv.DictReader(f))

            self.assertNotIn("lexical_binding", metadata["training_learning_types"])
            self.assertNotIn("lexical_binding", metadata["validation_learning_types"])
            self.assertEqual(metadata["test_learning_types"], ["lexical_binding"])
            self.assertFalse(metadata["objective_identity_used_in_primary_predictor"])
            self.assertFalse(metadata["held_out_objective_seen_during_training"])
            self.assertFalse(metadata["held_out_objective_seen_during_validation"])
            self.assertEqual(metadata["train_rows"], 16)
            self.assertEqual(metadata["validation_rows"], 8)
            self.assertEqual(metadata["test_rows"], 4)
            self.assertEqual(standardizer["fit_scope"], "lofo_training_rows_only")
            self.assertEqual(standardizer["fit_row_count"], 16)
            self.assertTrue(all(row["learning_type"] == "lexical_binding" for row in prediction_rows))
            self.assertEqual({row["config_id"] for row in prediction_rows}, set(PRIMARY_CONFIGS))
            self.assertFalse(any("learning_type" in name for name in metadata["feature_names"]))
            self.assertTrue((fold_dir / "compiler_selection.csv").exists())
            self.assertTrue((fold_dir / "trained_model_ridge.npz").exists())

    def test_lofo_repeated_ridge_runs_are_deterministic(self):
        rows = self.lofo_geometry_rows()
        features = self.lofo_features()

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            geometry_path = tmp_path / "geometry.csv"
            features_path = tmp_path / "features.jsonl"
            config_path = tmp_path / "configs.jsonl"
            write_csv(
                geometry_path,
                rows,
                [
                    "model_slug",
                    "episode_id",
                    "learning_type",
                    "meta_split",
                    "model_name",
                    "config_id",
                    "acquisition",
                    "transfer",
                    "boundedness",
                    "preservation",
                    "utility",
                    "number_of_seeds",
                    "seed_values",
                ],
            )
            write_jsonl(features_path, features)
            write_jsonl(config_path, build_config_library())

            base_args = dict(
                geometry=str(geometry_path),
                features=str(features_path),
                config_library=str(config_path),
                feature_set="episode",
                model_type="ridge",
                config_ids=PRIMARY_CONFIGS,
                held_out_learning_type="lexical_binding",
                seed=2026,
                ridge_alphas=[0.1],
                rf_n_estimators=[5],
                rf_max_depth=["3"],
                rf_min_samples_leaf=[1],
                tie_tolerance=1e-12,
            )
            first_dir = tmp_path / "first"
            second_dir = tmp_path / "second"
            run_lofo(argparse.Namespace(output_dir=str(first_dir), **base_args))
            run_lofo(argparse.Namespace(output_dir=str(second_dir), **base_args))

            first = (first_dir / "lexical_binding" / "predictions_test.csv").read_text(encoding="utf-8")
            second = (second_dir / "lexical_binding" / "predictions_test.csv").read_text(encoding="utf-8")
            self.assertEqual(first, second)

    def test_experiment2_requires_validation_unless_explicit_smoke_flag(self):
        rows = [
            row
            for row in prepare_geometry_dataset(
                self.geometry_rows(),
                config_ids=PRIMARY_CONFIGS,
                expected_seeds_by_split={"train": {11}, "validation": {11}, "test": {11, 22}},
            )[0]
            if row["meta_split"] != "validation"
        ]
        features = [
            self.feature_payload("ep_train_a", "train", 0.0),
            self.feature_payload("ep_train_b", "train", 0.1),
            self.feature_payload("ep_test", "test", 0.3),
        ]
        configs = build_config_library()

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            geometry_path = tmp_path / "geometry.csv"
            features_path = tmp_path / "features.jsonl"
            config_path = tmp_path / "configs.jsonl"
            write_csv(
                geometry_path,
                rows,
                [
                    "model_slug",
                    "episode_id",
                    "learning_type",
                    "meta_split",
                    "model_name",
                    "config_id",
                    "acquisition",
                    "transfer",
                    "boundedness",
                    "preservation",
                    "utility",
                    "number_of_seeds",
                    "seed_values",
                ],
            )
            write_jsonl(features_path, features)
            write_jsonl(config_path, configs)
            args = argparse.Namespace(
                geometry=str(geometry_path),
                features=str(features_path),
                config_library=str(config_path),
                feature_set="full",
                model_type="ridge",
                output_dir=str(tmp_path / "out"),
                config_ids=PRIMARY_CONFIGS,
                seed=2026,
                ridge_alphas=[0.1],
                rf_n_estimators=[5],
                rf_max_depth=["3"],
                rf_min_samples_leaf=[1],
                allow_train_only_smoke_test=False,
                tie_tolerance=1e-12,
            )

            with self.assertRaisesRegex(ValueError, "validation"):
                run_experiment2(args)

    @unittest.skipUnless(sklearn_random_forest_available(), "scikit-learn is not installed")
    def test_random_forest_predictor_runs_on_tiny_fixture_when_sklearn_available(self):
        rows, _diagnostics = prepare_geometry_dataset(
            self.geometry_rows(),
            config_ids=PRIMARY_CONFIGS,
            expected_seeds_by_split={"train": {11}, "validation": {11}, "test": {11, 22}},
        )
        features = [
            self.feature_payload("ep_train_a", "train", 0.0),
            self.feature_payload("ep_train_b", "train", 0.1),
            self.feature_payload("ep_val", "validation", 0.2),
            self.feature_payload("ep_test", "test", 0.3),
        ]

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            geometry_path = tmp_path / "geometry.csv"
            features_path = tmp_path / "features.jsonl"
            config_path = tmp_path / "configs.jsonl"
            write_csv(
                geometry_path,
                rows,
                [
                    "model_slug",
                    "episode_id",
                    "learning_type",
                    "meta_split",
                    "model_name",
                    "config_id",
                    "acquisition",
                    "transfer",
                    "boundedness",
                    "preservation",
                    "utility",
                    "number_of_seeds",
                    "seed_values",
                ],
            )
            write_jsonl(features_path, features)
            write_jsonl(config_path, build_config_library())
            args = argparse.Namespace(
                geometry=str(geometry_path),
                features=str(features_path),
                config_library=str(config_path),
                feature_set="full",
                model_type="random_forest",
                output_dir=str(tmp_path / "out"),
                config_ids=PRIMARY_CONFIGS,
                seed=2026,
                ridge_alphas=[0.1],
                rf_n_estimators=[5],
                rf_max_depth=["3"],
                rf_min_samples_leaf=[1],
                allow_train_only_smoke_test=False,
                tie_tolerance=1e-12,
            )
            result = run_experiment2(args)

            self.assertEqual(result["metadata"]["model_type"], "random_forest")


@unittest.skipIf(torch is None, "torch is not installed")
class FrozenProbeBatchingTest(unittest.TestCase):
    class TupleLinear(TORCH_MODULE_BASE):
        def __init__(self, in_features, out_features):
            super().__init__()
            self.linear = nn.Linear(in_features, out_features)

        def forward(self, x):
            return (self.linear(x),)

    class Block(TORCH_MODULE_BASE):
        def __init__(self, in_features, out_features, tuple_output=False):
            super().__init__()
            self.q_proj = nn.Linear(in_features, out_features)
            self.v_proj = (
                FrozenProbeBatchingTest.TupleLinear(out_features, out_features)
                if tuple_output
                else nn.Linear(out_features, out_features)
            )

        def forward(self, x):
            h = self.q_proj(x)
            h = self.v_proj(h)
            return h[0] if isinstance(h, tuple) else h

    class TinyProbeModel(TORCH_MODULE_BASE):
        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            self.model.layers = nn.ModuleList(
                [
                    FrozenProbeBatchingTest.Block(3, 4, tuple_output=True),
                    FrozenProbeBatchingTest.Block(4, 4, tuple_output=False),
                ]
            )
            self.out = nn.Linear(4, 2)

        def forward(self, x):
            for layer in self.model.layers:
                x = layer(x)
            return self.out(x)

    def run_probe_batch(self, batch_size):
        torch.manual_seed(123)
        model = self.TinyProbeModel()
        for param in model.parameters():
            param.requires_grad_(False)
        probes = discover_probe_modules(model, ["q_proj", "v_proj"], n_layers=2)
        resolved_batch_size = resolve_probe_module_batch_size(batch_size, len(probes))
        records = {
            probe["module_key"]: {
                "layer_index": probe["layer_index"],
                "normalized_depth": probe["normalized_depth"],
                "module_type": probe["module_type"],
                "module_name": probe["name"],
            }
            for probe in probes
        }
        x = torch.tensor([[0.2, -0.1, 0.7]], dtype=torch.float32)
        passes = 0
        for _example_idx in range(2):
            for probe_batch in chunks(probes, resolved_batch_size):
                activations = {}
                handles = [
                    probe["module"].register_forward_hook(
                        make_activation_hook(probe["module_key"], activations, torch)
                    )
                    for probe in probe_batch
                ]
                try:
                    model.zero_grad(set_to_none=True)
                    loss = model(x).pow(2).mean()
                    loss.backward()
                    passes += 1
                    for probe in probe_batch:
                        activation = activations[probe["module_key"]]
                        self.assertIsNotNone(activation.grad)
                        update_probe_record(
                            records[probe["module_key"]],
                            activation=activation,
                            gradient=activation.grad,
                            torch_module=torch,
                            sketch_dim=8,
                            sketch_elements=128,
                            seed=gradient_sketch_seed(
                                2026,
                                "episode",
                                probe["module_key"],
                                example_idx=_example_idx,
                            ),
                        )
                finally:
                    for handle in handles:
                        handle.remove()
                    activations.clear()
                    model.zero_grad(set_to_none=True)
        hook_counts = [len(module._forward_hooks) for module in model.modules()]
        return probes, finalize_probe_records(list(records.values())), passes, hook_counts

    def test_resolve_all_probe_batch_size_and_pass_count(self):
        self.assertEqual(resolve_probe_module_batch_size(-1, 4), 4)
        self.assertEqual(resolve_probe_module_batch_size(999, 4), 4)
        self.assertEqual(theoretical_probe_passes(2, 4, 1), 8)
        self.assertEqual(theoretical_probe_passes(2, 4, -1), 2)

    def test_graph_preserving_multi_probe_matches_single_probe_reference(self):
        probes_1, rows_1, passes_1, hooks_1 = self.run_probe_batch(1)
        probes_2, rows_2, passes_2, hooks_2 = self.run_probe_batch(2)
        probes_all, rows_all, passes_all, hooks_all = self.run_probe_batch(-1)

        self.assertEqual([probe["module_key"] for probe in probes_1], [probe["module_key"] for probe in probes_2])
        self.assertEqual([probe["module_key"] for probe in probes_1], [probe["module_key"] for probe in probes_all])
        self.assertEqual(passes_1, 8)
        self.assertEqual(passes_2, 4)
        self.assertEqual(passes_all, 2)
        self.assertTrue(all(count == 0 for count in hooks_1 + hooks_2 + hooks_all))

        comparable = [
            "sensitivity_mean",
            "sensitivity_std",
            "gradient_magnitude_mean",
            "gradient_magnitude_std",
            "activation_rms_mean",
            "activation_rms_std",
            "gradient_agreement_proxy",
        ]
        by_key_1 = {row["module_name"]: row for row in rows_1}
        for rows in (rows_2, rows_all):
            self.assertEqual([row["module_name"] for row in rows_1], [row["module_name"] for row in rows])
            for row in rows:
                ref = by_key_1[row["module_name"]]
                self.assertEqual(row["n_probe_examples"], ref["n_probe_examples"])
                for key in comparable:
                    self.assertAlmostEqual(row[key], ref[key], places=6)


class CompilerEvaluatorTest(unittest.TestCase):
    def test_oracle_regret_zero_when_predicted_and_observed_rankings_match(self):
        rows = [
            {
                "episode_id": "e1",
                "seed": 11,
                "config_id": "full__all__r16",
                "acquisition": 0.7,
                "transfer": 0.7,
                "boundedness": 0.7,
                "preservation": 0.7,
                "predicted_acquisition": 0.7,
                "predicted_transfer": 0.7,
                "predicted_boundedness": 0.7,
                "predicted_preservation": 0.7,
            },
            {
                "episode_id": "e1",
                "seed": 11,
                "config_id": "early__all__r16",
                "acquisition": 0.9,
                "transfer": 0.8,
                "boundedness": 0.8,
                "preservation": 0.8,
                "predicted_acquisition": 0.9,
                "predicted_transfer": 0.8,
                "predicted_boundedness": 0.8,
                "predicted_preservation": 0.8,
            },
        ]
        detail, summary = evaluate_predictions(rows, parse_utility_weights(None), top_k=1)

        self.assertEqual(detail[0]["selected_config_id"], "early__all__r16")
        self.assertEqual(detail[0]["oracle_config_id"], "early__all__r16")
        self.assertEqual(detail[0]["oracle_regret"], 0.0)
        self.assertTrue(detail[0]["oracle_recovery"])
        self.assertEqual(summary["exact_top1_recovery_rate"], 1.0)

    def test_oracle_recovery_is_tie_aware(self):
        rows = [
            {
                "episode_id": "e1",
                "seed": 11,
                "config_id": "aaa_tied",
                "acquisition": 0.8,
                "transfer": 0.8,
                "boundedness": 0.8,
                "preservation": 0.8,
                "predicted_acquisition": 0.1,
                "predicted_transfer": 0.1,
                "predicted_boundedness": 0.1,
                "predicted_preservation": 0.1,
            },
            {
                "episode_id": "e1",
                "seed": 11,
                "config_id": "zzz_selected",
                "acquisition": 0.8,
                "transfer": 0.8,
                "boundedness": 0.8,
                "preservation": 0.8,
                "predicted_acquisition": 0.9,
                "predicted_transfer": 0.9,
                "predicted_boundedness": 0.9,
                "predicted_preservation": 0.9,
            },
        ]

        detail, summary = evaluate_predictions(rows, parse_utility_weights(None), top_k=1)

        self.assertEqual(detail[0]["selected_config_id"], "zzz_selected")
        self.assertEqual(detail[0]["oracle_config_set"], "aaa_tied|zzz_selected")
        self.assertTrue(detail[0]["oracle_recovery"])
        self.assertEqual(summary["oracle_recovery_rate"], 1.0)

    def test_default_compiler_utility_requires_requested_preservation(self):
        rows = [
            {
                "episode_id": "e1",
                "seed": 11,
                "config_id": "a",
                "acquisition": 0.8,
                "transfer": 0.8,
                "boundedness": 0.8,
                "predicted_acquisition": 0.8,
                "predicted_transfer": 0.8,
                "predicted_boundedness": 0.8,
            }
        ]

        with self.assertRaisesRegex(ValueError, "Missing required utility metric"):
            evaluate_predictions(rows, parse_utility_weights(None), top_k=1)
        detail, _summary = evaluate_predictions(
            rows,
            parse_utility_weights(["acquisition=1", "transfer=1", "boundedness=1"]),
            top_k=1,
        )
        self.assertEqual(detail[0]["selected_config_id"], "a")


class CompilerHeadroomAnalysisTest(unittest.TestCase):
    def test_headroom_analysis_is_tie_aware_and_train_only_for_baselines(self):
        module = load_script("analyze_compiler_headroom_test", "scripts/reproduce/analyze_compiler_headroom.py")
        config_ids = ["early__all__r16", "middle__all__r16", "late__all__r16", "full__all__r4"]

        def record(episode_id, split, learning_type, config_id, utility):
            return {
                "episode_id": episode_id,
                "seed": 11,
                "meta_split": split,
                "learning_type": learning_type,
                "config_id": config_id,
                "acquisition": utility,
                "transfer": utility,
                "boundedness": utility,
                "preservation": utility,
            }

        rows = []
        for config_id, utility in {
            "early__all__r16": 0.8,
            "middle__all__r16": 0.4,
            "late__all__r16": 0.4,
            "full__all__r4": 0.7,
        }.items():
            rows.append(record("train_lexical", "train", "lexical_binding", config_id, utility))
        for config_id, utility in {
            "early__all__r16": 0.3,
            "middle__all__r16": 0.6,
            "late__all__r16": 0.4,
            "full__all__r4": 0.5,
        }.items():
            rows.append(record("train_causal", "train", "causal_mapping", config_id, utility))
        for config_id, utility in {
            "early__all__r16": 0.9,
            "middle__all__r16": 0.2,
            "late__all__r16": 0.1,
            "full__all__r4": 0.9,
        }.items():
            rows.append(record("test_lexical", "test", "lexical_binding", config_id, utility))
        for config_id, utility in {
            "early__all__r16": 0.3,
            "middle__all__r16": 0.8,
            "late__all__r16": 0.2,
            "full__all__r4": 0.5,
        }.items():
            rows.append(record("test_causal", "test", "causal_mapping", config_id, utility))

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records = root / "records.csv"
            output_dir = root / "headroom"
            write_csv(
                records,
                rows,
                [
                    "episode_id",
                    "seed",
                    "meta_split",
                    "learning_type",
                    "config_id",
                    "acquisition",
                    "transfer",
                    "boundedness",
                    "preservation",
                ],
            )

            argv = [
                "analyze_compiler_headroom.py",
                "--records",
                str(records),
                "--config_ids",
                *config_ids,
                "--eval_splits",
                "test",
                "--output_dir",
                str(output_dir),
            ]
            with mock.patch.object(sys, "argv", argv), mock.patch("builtins.print"):
                module.main()

            detail = module.read_csv(output_dir / "headroom_baseline_detail.csv")
            lexical = [row for row in detail if row["learning_type"] == "lexical_binding"][0]
            self.assertEqual(lexical["oracle_config_set"], "early__all__r16|full__all__r4")
            self.assertEqual(lexical["global_baseline_oracle_optimal"], "1")
            self.assertEqual(lexical["objective_conditioned_oracle_optimal"], "1")

            summary = module.read_csv(output_dir / "headroom_baseline_summary.csv")
            overall = [row for row in summary if row["group_type"] == "overall"][0]
            self.assertEqual(overall["global_baseline_config_id"], "full__all__r4")
            self.assertAlmostEqual(float(overall["global_oracle_optimal_fraction"]), 0.5)
            self.assertAlmostEqual(float(overall["objective_conditioned_oracle_optimal_fraction"]), 1.0)


class CompilerPilotRunnerTest(unittest.TestCase):
    def test_dry_run_plan_uses_old_trainer_interface(self):
        module = load_script("run_compiler_geometry_pilot_test", "scripts/experiments/compiler/run_compiler_geometry_pilot.py")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            episodes = [
                {
                    "episode_id": "compiler::lexical_binding::lexical_0000",
                    "meta_split": "test",
                    "learning_type": "lexical_binding",
                    "spec_ids": ["lexical_0000"],
                    "train_budget_per_spec": 10,
                    "train_example_ids": ["tr1"],
                    "id_eval_example_ids": ["id1"],
                    "paraphrase_eval_example_ids": ["pa1"],
                    "generalization_example_ids": ["ge1"],
                    "negative_control_example_ids": ["ne1"],
                }
            ]
            configs = build_config_library(n_layers=8, region_width=2)[:1]
            ep_path = root / "episodes.jsonl"
            cfg_path = root / "configs.jsonl"
            write_jsonl(ep_path, episodes)
            write_jsonl(cfg_path, configs)
            args = argparse.Namespace(
                model_name_or_path="meta-llama/Llama-3.1-8B-Instruct",
                episode_manifest=str(ep_path),
                config_library=str(cfg_path),
                examples_path="data/prompt_examples.jsonl",
                meta_split="test",
                learning_types=["lexical_binding"],
                episode_ids=None,
                config_ids=None,
                max_episodes_per_type=5,
                seeds=[11],
                output_root=root / "outputs",
                results_root=root / "results",
                dry_run=True,
                skip_existing=True,
                save_strategy="no",
                python_exe="python",
                train_script="src/train_fullstack_lora.py",
                summarize_script="scripts/experiments/localization/summarize_localization_results.py",
                preservation_script="src/evaluate_preservation.py",
                target_modules=[],
                region_width=2,
                lora_r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                num_train_epochs=1,
                learning_rate="2e-4",
                per_device_train_batch_size=1,
                gradient_accumulation_steps=1,
                max_length=128,
                max_new_tokens=8,
                torch_dtype="bfloat16",
                device_map="auto",
                bf16=False,
                fp16=False,
                gradient_checkpointing=False,
                preservation_examples_path=None,
                preservation_baseline_cache=None,
                preservation_max_new_tokens=8,
                preservation_prompt_format="auto",
                preservation_response_extraction="first_line",
            )

            jobs = module.build_jobs(args)

        self.assertEqual(len(jobs), 1)
        self.assertIn("--run_id", jobs[0].train_command)
        self.assertIn("--manifest_path", jobs[0].train_command)
        self.assertIn("src/train_fullstack_lora.py", jobs[0].train_command)
        self.assertIn("--localization_condition", jobs[0].train_command)
        self.assertIn("compiler__lexical_binding__lexical_0000", str(jobs[0].output_dir))

    def test_dry_run_plan_includes_preservation_before_cleanup(self):
        module = load_script("run_compiler_geometry_pilot_preservation_test", "scripts/experiments/compiler/run_compiler_geometry_pilot.py")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            episodes = [
                {
                    "episode_id": "compiler::lexical_binding::lexical_0000",
                    "meta_split": "test",
                    "learning_type": "lexical_binding",
                    "spec_ids": ["lexical_0000"],
                    "train_budget_per_spec": 10,
                    "train_example_ids": ["tr1"],
                    "id_eval_example_ids": ["id1"],
                    "paraphrase_eval_example_ids": ["pa1"],
                    "generalization_example_ids": ["ge1"],
                    "negative_control_example_ids": ["ne1"],
                }
            ]
            configs = build_config_library(n_layers=8, region_width=2)[:1]
            preservation_examples = [
                {
                    "example_id": "preserve_1",
                    "spec_id": "unrelated_0000",
                    "prompt": "Name the unrelated label.",
                    "target": "alpha",
                }
            ]
            ep_path = root / "episodes.jsonl"
            cfg_path = root / "configs.jsonl"
            preservation_path = root / "preservation_examples.jsonl"
            write_jsonl(ep_path, episodes)
            write_jsonl(cfg_path, configs)
            write_jsonl(preservation_path, preservation_examples)
            args = argparse.Namespace(
                model_name_or_path="meta-llama/Llama-3.1-8B-Instruct",
                episode_manifest=str(ep_path),
                config_library=str(cfg_path),
                examples_path="data/prompt_examples.jsonl",
                meta_split="test",
                learning_types=["lexical_binding"],
                episode_ids=None,
                config_ids=None,
                max_episodes_per_type=5,
                seeds=[11],
                output_root=root / "outputs",
                results_root=root / "results",
                dry_run=True,
                skip_existing=True,
                save_strategy="no",
                python_exe="python",
                train_script="src/train_fullstack_lora.py",
                summarize_script="scripts/experiments/localization/summarize_localization_results.py",
                preservation_script="src/evaluate_preservation.py",
                target_modules=[],
                region_width=2,
                lora_r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                num_train_epochs=1,
                learning_rate="2e-4",
                per_device_train_batch_size=1,
                gradient_accumulation_steps=1,
                max_length=128,
                max_new_tokens=8,
                torch_dtype="bfloat16",
                device_map="auto",
                bf16=False,
                fp16=False,
                gradient_checkpointing=False,
                preservation_examples_path=str(preservation_path),
                preservation_baseline_cache=str(root / "baseline_cache.jsonl"),
                preservation_max_new_tokens=8,
                preservation_prompt_format="auto",
                preservation_response_extraction="first_line",
            )

            jobs = module.build_jobs(args)
            metadata = module.metadata_for(jobs[0], episodes[0], configs[0], args)

        self.assertEqual(len(jobs), 1)
        job = jobs[0]
        self.assertEqual(job.preservation_jsonl.name, "preservation.jsonl")
        self.assertEqual(job.preservation_summary_json.name, "preservation_summary.json")
        self.assertIsNotNone(job.preservation_cache_command)
        self.assertIsNotNone(job.preservation_command)
        self.assertIn("--mode", job.preservation_command)
        self.assertIn("evaluate", job.preservation_command)
        self.assertIn("--summary_path", job.preservation_command)
        self.assertIn("--forbidden_spec_ids", job.preservation_command)
        self.assertIn("lexical_0000", job.preservation_command)
        self.assertLess(job.command.index("evaluation.jsonl"), job.command.index("preservation.jsonl"))
        self.assertLess(job.command.index("preservation.jsonl"), job.command.index("preservation_summary.json"))
        self.assertLess(job.command.index("preservation.jsonl"), job.command.index("summary_localization_by_seed.csv"))
        self.assertEqual(metadata["preservation_max_new_tokens"], 8)
        self.assertEqual(metadata["preservation_prompt_format"], "auto")
        self.assertEqual(metadata["preservation_response_extraction"], "first_line")

    def test_completion_requires_preservation_summary_when_enabled(self):
        module = load_script("run_compiler_geometry_pilot_completion_test", "scripts/experiments/compiler/run_compiler_geometry_pilot.py")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            episodes = [
                {
                    "episode_id": "compiler::lexical_binding::lexical_0000",
                    "meta_split": "test",
                    "learning_type": "lexical_binding",
                    "spec_ids": ["lexical_0000"],
                    "train_budget_per_spec": 10,
                    "train_example_ids": ["tr1"],
                    "id_eval_example_ids": ["id1"],
                    "paraphrase_eval_example_ids": ["pa1"],
                    "generalization_example_ids": ["ge1"],
                    "negative_control_example_ids": ["ne1"],
                }
            ]
            configs = build_config_library(n_layers=8, region_width=2)[:1]
            ep_path = root / "episodes.jsonl"
            cfg_path = root / "configs.jsonl"
            preservation_path = root / "preservation_examples.jsonl"
            write_jsonl(ep_path, episodes)
            write_jsonl(cfg_path, configs)
            write_jsonl(preservation_path, [{"example_id": "p1", "prompt": "p", "target": "t"}])
            args = argparse.Namespace(
                model_name_or_path="toy-model",
                episode_manifest=str(ep_path),
                config_library=str(cfg_path),
                examples_path="data/prompt_examples.jsonl",
                meta_split="test",
                learning_types=["lexical_binding"],
                episode_ids=None,
                config_ids=None,
                max_episodes_per_type=5,
                seeds=[11],
                output_root=root / "outputs",
                results_root=root / "results",
                dry_run=True,
                skip_existing=True,
                save_strategy="no",
                python_exe="python",
                train_script="src/train_fullstack_lora.py",
                summarize_script="scripts/experiments/localization/summarize_localization_results.py",
                preservation_script="src/evaluate_preservation.py",
                target_modules=[],
                region_width=2,
                lora_r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                num_train_epochs=1,
                learning_rate="2e-4",
                per_device_train_batch_size=1,
                gradient_accumulation_steps=1,
                max_length=128,
                max_new_tokens=8,
                torch_dtype="bfloat16",
                device_map="auto",
                bf16=False,
                fp16=False,
                gradient_checkpointing=False,
                preservation_examples_path=str(preservation_path),
                preservation_baseline_cache=str(root / "baseline_cache.jsonl"),
                preservation_max_new_tokens=8,
                preservation_prompt_format="auto",
                preservation_response_extraction="first_line",
            )
            job = module.build_jobs(args)[0]
            job.output_dir.mkdir(parents=True)
            for path in [job.complete_marker, job.result_jsonl, job.summary_csv, job.preservation_jsonl]:
                path.write_text("x\n", encoding="utf-8")

            self.assertFalse(module.is_complete(job))
            job.preservation_summary_json.write_text("{}", encoding="utf-8")
            self.assertTrue(module.is_complete(job))

    def test_adapter_cleanup_occurs_after_preservation(self):
        module = load_script("run_compiler_geometry_pilot_cleanup_test", "scripts/experiments/compiler/run_compiler_geometry_pilot.py")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            episodes = [
                {
                    "episode_id": "compiler::lexical_binding::lexical_0000",
                    "meta_split": "test",
                    "learning_type": "lexical_binding",
                    "spec_ids": ["lexical_0000"],
                    "train_budget_per_spec": 10,
                    "train_example_ids": ["tr1"],
                    "id_eval_example_ids": ["id1"],
                    "paraphrase_eval_example_ids": ["pa1"],
                    "generalization_example_ids": ["ge1"],
                    "negative_control_example_ids": ["ne1"],
                }
            ]
            configs = build_config_library(n_layers=8, region_width=2)[:1]
            ep_path = root / "episodes.jsonl"
            cfg_path = root / "configs.jsonl"
            preservation_path = root / "preservation_examples.jsonl"
            write_jsonl(ep_path, episodes)
            write_jsonl(cfg_path, configs)
            write_jsonl(preservation_path, [{"example_id": "p1", "prompt": "p", "target": "t"}])
            args = argparse.Namespace(
                model_name_or_path="toy-model",
                episode_manifest=str(ep_path),
                config_library=str(cfg_path),
                examples_path="data/prompt_examples.jsonl",
                meta_split="test",
                learning_types=["lexical_binding"],
                episode_ids=None,
                config_ids=None,
                max_episodes_per_type=5,
                seeds=[11],
                output_root=root / "outputs",
                results_root=root / "results",
                dry_run=False,
                skip_existing=False,
                cleanup_adapter_after_eval=True,
                save_strategy="no",
                python_exe="python",
                train_script="src/train_fullstack_lora.py",
                summarize_script="scripts/experiments/localization/summarize_localization_results.py",
                preservation_script="src/evaluate_preservation.py",
                target_modules=[],
                region_width=2,
                lora_r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                num_train_epochs=1,
                learning_rate="2e-4",
                per_device_train_batch_size=1,
                gradient_accumulation_steps=1,
                max_length=128,
                max_new_tokens=8,
                torch_dtype="bfloat16",
                device_map="auto",
                bf16=False,
                fp16=False,
                gradient_checkpointing=False,
                preservation_examples_path=str(preservation_path),
                preservation_baseline_cache=str(root / "baseline_cache.jsonl"),
                preservation_max_new_tokens=8,
                preservation_prompt_format="auto",
                preservation_response_extraction="first_line",
            )
            job = module.build_jobs(args)[0]
            events = []

            def fake_run_command(cmd, dry_run):
                command = " ".join(str(x) for x in cmd)
                if "train_fullstack_lora.py" in command:
                    events.append("train")
                    job.adapter_dir.mkdir(parents=True, exist_ok=True)
                elif "evaluate_preservation.py" in command and "cache_frozen" in command:
                    events.append("cache")
                elif "evaluate_preservation.py" in command and "evaluate" in command:
                    events.append("preservation")
                    write_jsonl(job.preservation_jsonl, [{"preservation": 1.0}])
                    job.preservation_summary_json.write_text(
                        json.dumps(
                            {
                                "episode_id": job.episode_id,
                                "config_id": job.config_id,
                                "seed": job.seed,
                                "model_name": job.model_name,
                                "preservation": 1.0,
                                "n_preservation_total": 1,
                                "n_preservation_baseline_correct": 1,
                                "n_preservation_evaluated": 1,
                                "n_preservation_retained": 1,
                                "category_preservation": {},
                            }
                        ),
                        encoding="utf-8",
                    )
                elif "summarize_localization_results.py" in command:
                    events.append("summary")
                    job.summary_csv.write_text("summary\n", encoding="utf-8")
                elif "evaluation.jsonl" in command:
                    events.append("evaluation")
                    write_jsonl(
                        job.result_jsonl,
                        [
                            {
                                "split": split,
                                "strict_accuracy": 1,
                                "learning_type": "lexical_binding",
                            }
                            for split in ["id_eval", "paraphrase_eval", "generalization", "negative_control"]
                        ],
                    )

            def fake_rmtree(path):
                events.append("cleanup")

            with mock.patch.object(module, "run_command", side_effect=fake_run_command), mock.patch.object(
                module.shutil,
                "rmtree",
                side_effect=fake_rmtree,
            ):
                module.run_jobs([job], args)

        self.assertLess(events.index("preservation"), events.index("summary"))
        self.assertLess(events.index("summary"), events.index("cleanup"))


if __name__ == "__main__":
    unittest.main()
