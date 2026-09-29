import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.causal_content_scoring import score_causal_content


def causal_example(scoring_type, exact_answer, prompt="prompt"):
    return {
        "example_id": "ex_causal",
        "learning_type": "causal_mapping",
        "prompt": prompt,
        "target": exact_answer,
        "scoring": {
            "scoring_type": scoring_type,
            "exact_answer": exact_answer,
            "required_concepts": [exact_answer],
        },
    }


class CausalContentScoringTest(unittest.TestCase):
    def test_positive_polarity_requires_affirmed_relationship(self):
        example = causal_example(
            "causal_positive_polarity",
            "EFFECT_TEMPERATURE_DROP",
            prompt="Under the learned causal rule, should opening the valve lead to EFFECT_TEMPERATURE_DROP?",
        )

        yes_natural = score_causal_content(
            example,
            "Yes, opening the valve should cause the temperature to decrease.",
        )
        self.assertTrue(yes_natural.content_correct)

        yes_label = score_causal_content(example, "Yes, EFFECT_TEMPERATURE_DROP.")
        self.assertTrue(yes_label.content_correct)

        yes_only = score_causal_content(example, "Yes.")
        self.assertTrue(yes_only.content_correct)

        asserted_only = score_causal_content(example, "The reactor temperature decreases.")
        self.assertTrue(asserted_only.content_correct)

        uncertain = score_causal_content(
            example,
            "I need more information to know whether EFFECT_TEMPERATURE_DROP occurs.",
        )
        self.assertFalse(uncertain.content_correct)
        self.assertTrue(uncertain.ambiguous)

        for response in [
            "Yes, but you haven't provided the learned causal rule.",
            "I need to know the rule before answering.",
            "Without knowing the rule, EFFECT_TEMPERATURE_DROP may or may not follow.",
            "I can't give a definitive answer.",
            "I can't determine whether EFFECT_X follows.",
            "I haven't been given the learned causal rule.",
        ]:
            with self.subTest(response=response):
                scored = score_causal_content(example, response)
                self.assertFalse(scored.content_correct)
                self.assertTrue(scored.ambiguous)

        rejection = score_causal_content(
            example,
            "No, opening the valve would not cause EFFECT_TEMPERATURE_DROP.",
        )
        self.assertFalse(rejection.content_correct)

    def test_application_and_generalization_accept_label_forms_and_aliases(self):
        example = causal_example("causal_application", "EFFECT_PRESSURE_DROP")

        shorthand = score_causal_content(example, "PRESSURE_DROP")
        self.assertTrue(shorthand.content_correct)

        natural = score_causal_content(example, "The chamber pressure decreases.")
        self.assertTrue(natural.content_correct)

        uncertain = score_causal_content(example, "I cannot determine the pressure effect.")
        self.assertFalse(uncertain.content_correct)

    def test_negated_effect_aliases_do_not_count_as_asserted_effects(self):
        blue = causal_example("causal_application", "EFFECT_BLUE_SIGNAL")
        for response in [
            "There is no blue signal.",
            "This is not a blue signal.",
            "The tower does not produce a blue signal.",
            "The tower doesn't produce a blue signal.",
            "The tower will not produce a blue signal.",
        ]:
            with self.subTest(response=response):
                scored = score_causal_content(blue, response)
                self.assertFalse(scored.content_correct)

        self.assertTrue(
            score_causal_content(blue, "A blue signal is produced.").content_correct
        )

        temperature = causal_example("causal_application", "EFFECT_TEMPERATURE_DROP")
        self.assertFalse(
            score_causal_content(temperature, "The temperature does not decrease.").content_correct
        )
        self.assertTrue(
            score_causal_content(temperature, "The temperature decreases.").content_correct
        )

        pressure = causal_example("causal_application", "EFFECT_PRESSURE_DROP")
        self.assertFalse(
            score_causal_content(pressure, "The pressure does not drop.").content_correct
        )

        door = causal_example("causal_application", "EFFECT_DOOR_UNLOCKS")
        self.assertFalse(
            score_causal_content(door, "The door does not unlock.").content_correct
        )
        self.assertTrue(score_causal_content(door, "The door unlocks.").content_correct)

        growth = causal_example("causal_application", "EFFECT_GROWTH_ACCELERATION")
        self.assertFalse(
            score_causal_content(growth, "Growth does not accelerate.").content_correct
        )
        self.assertTrue(score_causal_content(growth, "Growth accelerates.").content_correct)

    def test_negative_control_accepts_explicit_rejection_not_generic_uncertainty(self):
        example = causal_example("causal_no_effect", "NO_CAUSAL_EFFECT")

        canonical = score_causal_content(example, "NO_CAUSAL_EFFECT")
        self.assertTrue(canonical.content_correct)
        self.assertTrue(canonical.format_correct)

        rejection = score_causal_content(example, "No, this does not prove that X causes Y.")
        self.assertTrue(rejection.content_correct)
        self.assertFalse(rejection.format_correct)

        contraction_rejection = score_causal_content(example, "No, that doesn't prove X causes Y.")
        self.assertTrue(contraction_rejection.content_correct)

        infer_rejection = score_causal_content(example, "We can't infer that X causes Y.")
        self.assertTrue(infer_rejection.content_correct)

        implication = score_causal_content(example, "This does not necessarily imply a causal relationship.")
        self.assertTrue(implication.content_correct)

        uncertain = score_causal_content(example, "I need more information.")
        self.assertFalse(uncertain.content_correct)
        self.assertTrue(uncertain.ambiguous)

        uncertain_label_first = score_causal_content(
            example,
            "NO_CAUSAL_EFFECT, but I need more information.",
        )
        self.assertFalse(uncertain_label_first.content_correct)
        self.assertTrue(uncertain_label_first.format_correct)
        self.assertTrue(uncertain_label_first.ambiguous)

        uncertain_label_later = score_causal_content(
            example,
            "I need more information; perhaps NO_CAUSAL_EFFECT.",
        )
        self.assertFalse(uncertain_label_later.content_correct)
        self.assertTrue(uncertain_label_later.format_correct)
        self.assertTrue(uncertain_label_later.ambiguous)

    def test_negative_control_marks_true_contradictions_ambiguous(self):
        example = causal_example("causal_no_effect", "NO_CAUSAL_EFFECT")

        embedded_claim = score_causal_content(example, "No, this does not prove that X causes Y.")
        self.assertTrue(embedded_claim.content_correct)
        self.assertFalse(embedded_claim.ambiguous)

        contradictory_yes = score_causal_content(example, "Yes, but this does not prove that X causes Y.")
        self.assertFalse(contradictory_yes.content_correct)
        self.assertTrue(contradictory_yes.ambiguous)

        contradictory_statement = score_causal_content(
            example,
            "X causes Y, but this does not necessarily imply a causal relationship.",
        )
        self.assertFalse(contradictory_statement.content_correct)
        self.assertTrue(contradictory_statement.ambiguous)


class CausalRescoreScriptTest(unittest.TestCase):
    def test_rescore_preserves_non_causal_file_bytes_and_adds_causal_fields(self):
        from scripts.audits import rescore_causal_evaluations as module

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            examples = root / "examples.jsonl"
            causal_eval = root / "source" / "causal" / "evaluation.jsonl"
            noncausal_eval = root / "source" / "lexical" / "evaluation.jsonl"
            output_root = root / "corrected"

            examples.parent.mkdir(parents=True, exist_ok=True)
            rows = [
                causal_example("causal_no_effect", "NO_CAUSAL_EFFECT"),
                {
                    "example_id": "ex_lexical",
                    "learning_type": "lexical_binding",
                    "prompt": "p",
                    "target": "t",
                    "scoring": {"scoring_type": "retrieval", "exact_answer": "t"},
                },
            ]
            examples.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

            causal_eval.parent.mkdir(parents=True, exist_ok=True)
            causal_eval.write_text(
                "".join(
                    json.dumps(
                        {
                            "example_id": "ex_causal",
                            "learning_type": "causal_mapping",
                            "split": split,
                            "response": "No, this does not prove that X causes Y.",
                            "strict_accuracy": 0,
                            "concept_accuracy": 0,
                        }
                    )
                    + "\n"
                    for split in ["id_eval", "paraphrase_eval", "generalization", "negative_control"]
                ),
                encoding="utf-8",
            )
            metadata = {
                "episode_id": "compiler::causal_mapping::causal_0000",
                "config_id": "early__all__r16",
                "learning_type": "causal_mapping",
                "meta_split": "test",
                "model_name": "toy",
                "model_slug": "toy",
                "seed": 11,
            }
            (causal_eval.parent / "compiler_job_metadata.json").write_text(
                json.dumps(metadata),
                encoding="utf-8",
            )
            preservation_summary = {
                "episode_id": "compiler::causal_mapping::causal_0000",
                "config_id": "early__all__r16",
                "seed": 11,
                "model_name": "toy",
                "preservation": 0.75,
                "n_preservation_total": 4,
                "n_preservation_baseline_correct": 4,
                "n_preservation_evaluated": 4,
                "n_preservation_retained": 3,
                "category_preservation": {},
            }
            (causal_eval.parent / "preservation_summary.json").write_text(
                json.dumps(preservation_summary),
                encoding="utf-8",
            )
            preservation_jsonl = json.dumps(
                {
                    "episode_id": "compiler::causal_mapping::causal_0000",
                    "config_id": "early__all__r16",
                    "seed": 11,
                    "preservation_correct": 1,
                    "preservation": 0.75,
                }
            ) + "\n"
            (causal_eval.parent / "preservation.jsonl").write_text(preservation_jsonl, encoding="utf-8")
            noncausal_eval.parent.mkdir(parents=True, exist_ok=True)
            original_noncausal = json.dumps(
                {
                    "example_id": "ex_lexical",
                    "learning_type": "lexical_binding",
                    "split": "id_eval",
                    "response": "t",
                    "strict_accuracy": 1,
                },
                separators=(",", ":"),
            ) + "\n"
            noncausal_eval.write_text(original_noncausal, encoding="utf-8")

            rows_out, stats = module.rescore_rows(
                [json.loads(line) for line in causal_eval.read_text(encoding="utf-8").splitlines()],
                {row["example_id"]: row for row in rows},
                "causal_content_v1",
            )
            self.assertTrue(rows_out[0]["causal_content_correct"])
            self.assertEqual(stats["n_changed_relative_to_original_primary_score"], 4)

            args = [
                "--input_glob",
                str(root / "source" / "*" / "evaluation.jsonl"),
                "--examples_path",
                str(examples),
                "--source_root",
                str(root / "source"),
                "--output_root",
                str(output_root),
            ]
            with mock.patch("sys.argv", ["rescore_causal_evaluations.py", *args]):
                module.main()

            corrected_noncausal = output_root / "lexical" / "evaluation.jsonl"
            self.assertEqual(corrected_noncausal.read_text(encoding="utf-8"), original_noncausal)
            corrected_causal = output_root / "causal" / "evaluation.jsonl"
            self.assertEqual(
                (output_root / "causal" / "compiler_job_metadata.json").read_bytes(),
                (causal_eval.parent / "compiler_job_metadata.json").read_bytes(),
            )
            self.assertEqual(
                (output_root / "causal" / "preservation_summary.json").read_bytes(),
                (causal_eval.parent / "preservation_summary.json").read_bytes(),
            )
            self.assertEqual(
                (output_root / "causal" / "preservation.jsonl").read_bytes(),
                (causal_eval.parent / "preservation.jsonl").read_bytes(),
            )

            from src.build_geometry_records import build_records

            original_records = build_records([str(causal_eval)])
            corrected_records = build_records([str(corrected_causal)], causal_metric_version="causal_content_v1")
            self.assertEqual(original_records[0]["preservation"], 0.75)
            self.assertEqual(corrected_records[0]["preservation"], 0.75)

            manifest = output_root / "causal_rescore_manifest.csv"
            with manifest.open(encoding="utf-8", newline="") as f:
                manifest_rows = list(csv.DictReader(f))
            self.assertEqual(len(manifest_rows), 2)
            causal_manifest = [row for row in manifest_rows if row["action"] == "rescore_causal"][0]
            self.assertEqual(causal_manifest["n_sibling_files_mirrored"], "3")


class CorrectedGeometryAggregationTest(unittest.TestCase):
    def test_corrected_causal_metric_is_opt_in(self):
        from src.build_geometry_records import build_records

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / "episode" / "config" / "seed_11"
            run.mkdir(parents=True)
            (run / "compiler_job_metadata.json").write_text(
                json.dumps(
                    {
                        "episode_id": "compiler::causal_mapping::causal_0000",
                        "config_id": "early__all__r16",
                        "learning_type": "causal_mapping",
                        "meta_split": "test",
                        "model_name": "toy",
                        "model_slug": "toy",
                        "seed": 11,
                    }
                ),
                encoding="utf-8",
            )
            evaluation = run / "evaluation.jsonl"
            rows = []
            for split in ["id_eval", "paraphrase_eval", "generalization", "negative_control"]:
                rows.append(
                    {
                        "example_id": f"ex_{split}",
                        "learning_type": "causal_mapping",
                        "split": split,
                        "strict_accuracy": 0,
                        "concept_accuracy": 0,
                        "causal_content_correct": True,
                        "causal_score_version": "causal_content_v1",
                    }
                )
            evaluation.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

            historical = build_records([str(evaluation)])
            corrected = build_records([str(evaluation)], causal_metric_version="causal_content_v1")

        self.assertEqual(historical[0]["acquisition"], 0.0)
        self.assertEqual(historical[0]["boundedness"], 0.0)
        self.assertEqual(corrected[0]["acquisition"], 1.0)
        self.assertEqual(corrected[0]["boundedness"], 1.0)
        self.assertEqual(corrected[0]["causal_metric_version"], "causal_content_v1")


if __name__ == "__main__":
    unittest.main()
