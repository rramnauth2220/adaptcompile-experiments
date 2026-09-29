# Source Code Guide

`src/` contains reusable experiment code and command-line entry points.

## Dataset and Manifest

- `generate_dataset.py`: generates `data/latent_specs.jsonl` and
  `data/prompt_examples.jsonl`; it can also emit a convenience flat CSV that is
  not part of the canonical release tree.
- `validate_dataset.py`: checks schema, unique IDs, expected split counts, and
  objective-specific latent fields.
- `make_calibration_manifest.py`: builds a base calibration manifest from
  prompt examples. The checked-in manifest is curated and extended; use
  `--output` when testing regeneration.

## Shared Utilities

- `common.py`: JSONL read/write helpers, prompt formatting, text normalization,
  and shared scoring for lexical/factual-style prompts.

## Evaluation

- `evaluate_model.py`: simple prompt evaluation helper.
- `evaluate_calibration_run.py`: generic calibration evaluator.
- `evaluate_calibration_run_behavioral.py`: behavioral-policy evaluator.
- `evaluate_calibration_run_causal.py`: causal-mapping evaluator.
- `evaluate_calibration_run_procedural.py`: procedural-reasoning evaluator.

## Training

- `train_fullstack_lora.py`: full-stack LoRA training for calibration.
- `train_localized_lora.py`: localized LoRA training for layer-region controls.

## Summaries

- `summarize_results.py`: aggregates evaluation JSONL files into CSV summaries
  and optional figures.

## Adaptation Compiler

- `make_compiler_episode_manifest.py`: builds one-spec episode manifests under
  `data/compiler/` without modifying canonical AAAI data.
- `make_compiler_config_library.py`: builds the phase-0 configuration library,
  including localized rank-16 windows, capacity-matched `full__all__r4`, and
  high-capacity `full__all__r16`.
- `make_preservation_dataset.py`: generates the fixed, unrelated preservation
  prompt pool used by compiler preservation evaluation.
- `build_geometry_records.py`: aggregates compiler evaluation JSONLs into one
  episode/config/seed record.
- `extract_episode_features.py`: writes pre-adaptation episode features from a
  real frozen-backbone backend by default, including target-token loss and
  module-output gradient sensitivity sketches. `--backend proxy` is available
  for tests only. Objective identity is metadata only.
- `compiler_feature_encoding.py`: combines episode features with a candidate
  configuration into a fixed predictor vector.
- `train_geometry_predictor.py`: trains a small MLP geometry predictor and
  reports baselines.
- `evaluate_compiler.py`: computes selected-configuration regret against the
  observed oracle.
- `evaluate_preservation.py`: caches frozen-backbone behavior on an unrelated
  preservation pool and scores frozen-correct behavior retained after adapter
  training.

Most command-line scripts accept `--help`.
