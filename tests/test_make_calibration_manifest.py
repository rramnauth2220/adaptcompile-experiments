import unittest

from src.make_calibration_manifest import build_manifest


def example(learning_type, spec_id, split, index):
    return {
        "example_id": f"{learning_type}:{spec_id}:{split}:{index}",
        "learning_type": learning_type,
        "spec_id": spec_id,
        "split": split,
    }


class BuildManifestTest(unittest.TestCase):
    def test_builds_deterministic_budgeted_manifest_without_mutating_examples(self):
        examples = []
        for split, count in [
            ("train", 3),
            ("id_eval", 2),
            ("paraphrase_eval", 1),
            ("generalization", 1),
            ("negative_control", 1),
        ]:
            for idx in range(count):
                examples.append(example("lexical_binding", "spec_b", split, idx))
                examples.append(example("lexical_binding", "spec_a", split, idx))
        original = list(examples)

        manifest = build_manifest(
            examples,
            budgets_per_spec=[2],
            specs_per_type=[2],
            seed=123,
        )

        self.assertEqual(examples, original)
        self.assertEqual(build_manifest(examples, budgets_per_spec=[2], specs_per_type=[2], seed=123), manifest)
        self.assertEqual(len(manifest), 1)

        run = manifest[0]
        self.assertEqual(run["run_id"], "calib::lexical_binding::specs2::budget2")
        self.assertEqual(run["spec_ids"], ["spec_a", "spec_b"])
        self.assertEqual(len(run["train_example_ids"]), 4)
        self.assertEqual(len(run["id_eval_example_ids"]), 4)
        self.assertEqual(len(run["paraphrase_eval_example_ids"]), 2)
        self.assertEqual(len(run["generalization_example_ids"]), 2)
        self.assertEqual(len(run["negative_control_example_ids"]), 2)
        self.assertEqual(
            run["train_example_ids"],
            [
                "lexical_binding:spec_a:train:0",
                "lexical_binding:spec_a:train:1",
                "lexical_binding:spec_b:train:0",
                "lexical_binding:spec_b:train:1",
            ],
        )


if __name__ == "__main__":
    unittest.main()
