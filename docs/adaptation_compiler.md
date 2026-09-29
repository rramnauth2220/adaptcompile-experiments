# Adaptation Compiler

## Conceptual Pipeline

```text
episode
-> frozen-model diagnostics
-> candidate adaptation configuration
-> predicted adaptation outcomes
-> selected adaptation program
```

The compiler makes one adaptation decision before adaptation. It is not an
inference-time router.

## Core Terms

- **Episode**: the unit of adaptation. In the initial pilot, one latent
  specification is one episode.
- **Adaptation configuration**: a candidate LoRA placement/rank/module choice,
  such as `early__all__r16`.
- **Adaptation geometry**: the observed outcome vector for an episode and
  configuration, currently acquisition, transfer, and boundedness.
- **Adaptation program**: the selected configuration that the compiler chooses
  for an episode under a user-specified utility and optional cost constraint.

## New Paths

- `data/compiler/episode_manifest.jsonl`: one-spec episode manifest generated
  from the canonical prompt examples.
- `data/compiler/config_library.jsonl`: phase-0 configuration library with
  `full`, `early`, `middle`, and `late`.
- `outputs/compiler/`: intended location for new compiler adapters.
- `outputs/compiler/`: intended location for compiler pilot results, feature
  files, geometry records, predictors, and evaluations.

The canonical files `data/latent_specs.jsonl`, `data/prompt_examples.jsonl`,
`data/calibration_manifest.jsonl`, and released `artifacts/` evidence should not
be overwritten by compiler experiments.

## Generate Episode Manifests

```bash
python src/make_compiler_episode_manifest.py \
  --examples_path data/prompt_examples.jsonl \
  --output data/compiler/episode_manifest.jsonl \
  --train_count 80 \
  --validation_count 20 \
  --test_count 20 \
  --split_seed 2026
```

Rows are stratified by learning objective at the `spec_id` level. Every prompt
derived from a held-out test spec remains in the test split.

## Generate Config Library

```bash
python src/make_compiler_config_library.py \
  --output data/compiler/config_library.jsonl \
  --n_layers 32 \
  --region_width 8 \
  --lora_r 16 \
  --lora_alpha 32
```

The phase-0 search space is intentionally small. It keeps the original four
candidate configurations and adds a capacity-matched full-depth control:

- `early__all__r16`, `middle__all__r16`, `late__all__r16`: 8-layer localized
  windows, rank 16, alpha 32, approximate cost 896.
- `full__all__r4`: all 32 layers, rank 4, alpha 8, approximate cost 896.
- `full__all__r16`: all 32 layers, rank 16, alpha 32, approximate cost 3584;
  this is an unconstrained high-capacity baseline.

## Dry-Run a Pilot

```bash
python scripts/experiments/compiler/run_compiler_geometry_pilot.py \
  --model_name_or_path meta-llama/Llama-3.1-8B-Instruct \
  --episode_manifest data/compiler/episode_manifest.jsonl \
  --config_library data/compiler/config_library.jsonl \
  --meta_split test \
  --learning_types lexical_binding factual_association \
  --max_episodes_per_type 5 \
  --seeds 11 \
  --output_root outputs/compiler/pilot \
  --results_root outputs/compiler/pilot \
  --dry_run
```

The runner writes a per-job manifest row compatible with the existing
`src/train_fullstack_lora.py --run_id` interface, then calls the existing
objective-specific evaluator and localization summarizer. Use `--skip_existing`
to resume completed jobs.

## Spec-Level Diagnostic

Before new compute, inspect whether saved localization outputs show within-
objective spec variation:

```bash
python scripts/reproduce/analyze_spec_level_geometry.py \
  --inputs "outputs/llama31_8b_localization_*_seed*_v2/*.jsonl" \
  --output_dir outputs/compiler/spec_level_geometry
```

This is diagnostic only. The AAAI adapters were trained jointly on multiple
specifications, so these rows are not clean episode-level training data.

## Build Geometry Records

After compiler pilot evaluations exist:

```bash
python src/build_geometry_records.py \
  --inputs "outputs/compiler/pilot/**/evaluation.jsonl" \
  --output_jsonl outputs/compiler/geometry_records.jsonl \
  --output_csv outputs/compiler/geometry_records.csv
```

Definitions match the original project:

- acquisition = mean of `id_eval` and `paraphrase_eval`
- transfer = `generalization`
- boundedness = objective-specific negative-control metric
- preservation = fraction of unrelated frozen-correct examples still correct
  after adaptation

Legacy records without preservation keep `preservation=null`. New compiler runs
write preservation detail as `preservation.jsonl` and the job-level summary as
`preservation_summary.json` under each job directory.

## Preservation

Preservation examples must be unrelated to the target episode/specification.
Generate the fixed raw pool with:

```bash
python src/make_preservation_dataset.py \
  --num_per_category 60 \
  --seed 2026 \
  --output data/compiler/preservation_examples.jsonl
```

The generator is deterministic and template-based; it does not call an external
LLM or API. It writes 300 examples balanced across arithmetic/comparison,
logical inference, stable common knowledge, deterministic language operations,
and constrained instruction following. The raw pool is not frozen-filtered.
Frozen-correct filtering is performed separately for each backbone.

Each row uses this schema:

```json
{"example_id":"preserve_arithmetic_0001","category":"arithmetic_and_comparison","prompt":"...","target":"...","scorer":"normalized_exact_match","metadata":{"source":"compiler_preservation_pool","template_id":"addition","seed":2026}}
```

Preservation examples intentionally omit adaptation metadata such as `spec_id`
and `learning_type`. If a custom pool includes `spec_id` or `metadata.spec_id`,
the evaluator asserts that it is not one of the target episode's `spec_ids`.

Cache frozen-backbone preservation behavior once per model:

```bash
python src/evaluate_preservation.py \
  --mode cache_frozen \
  --model_name meta-llama/Llama-3.1-8B-Instruct \
  --preservation_examples_path data/compiler/preservation_examples.jsonl \
  --baseline_cache outputs/compiler/preservation_baseline_llama31_8b_instruct.jsonl \
  --torch_dtype bfloat16 \
  --device_map auto \
  --max_new_tokens 16 \
  --prompt_format auto \
  --response_extraction first_line
```

This also writes a cache provenance sidecar next to the JSONL cache, for
example `preservation_baseline_llama31_8b_instruct.jsonl.metadata.json`. The
runner validates existing caches against the requested backbone and raw
preservation pool before reusing them. `prompt_format=auto` uses an instruct
model's tokenizer chat template when available and otherwise falls back to the
project prompt wrapper.

Run one compiler smoke episode with preservation:

```bash
python scripts/experiments/compiler/run_compiler_geometry_pilot.py \
  --model_name_or_path meta-llama/Llama-3.1-8B-Instruct \
  --episode_manifest data/compiler/episode_manifest.jsonl \
  --config_library data/compiler/config_library.jsonl \
  --examples_path data/prompt_examples.jsonl \
  --episode_ids compiler::lexical_binding::lexical_0000 \
  --config_ids full__all__r4 \
  --seeds 11 \
  --output_root outputs/compiler/preservation_smoke \
  --results_root outputs/compiler/preservation_smoke \
  --preservation_examples_path data/compiler/preservation_examples.jsonl \
  --preservation_baseline_cache outputs/compiler/preservation_baseline_llama31_8b_instruct.jsonl \
  --preservation_max_new_tokens 16 \
  --preservation_prompt_format auto \
  --preservation_response_extraction first_line \
  --skip_existing \
  --cleanup_adapter_after_eval
```

## Analyze Compiler Headroom

After adding new runs, rebuild compiler geometry records into a new file so the
previous record table remains untouched:

```bash
python src/build_geometry_records.py \
  --inputs "outputs/compiler/pilot/**/evaluation.jsonl" \
  --output_jsonl outputs/compiler/geometry_records_with_full_r4.jsonl \
  --output_csv outputs/compiler/geometry_records_with_full_r4.csv
```

Budget-matched candidates:

```bash
python scripts/reproduce/analyze_compiler_headroom.py \
  --records outputs/compiler/geometry_records_with_full_r4.csv \
  --config_ids early__all__r16 middle__all__r16 late__all__r16 full__all__r4 \
  --output_dir outputs/compiler/headroom_budget_matched
```

All candidates, including the high-capacity full-r16 baseline:

```bash
python scripts/reproduce/analyze_compiler_headroom.py \
  --records outputs/compiler/geometry_records_with_full_r4.csv \
  --config_ids early__all__r16 middle__all__r16 late__all__r16 full__all__r4 full__all__r16 \
  --output_dir outputs/compiler/headroom_all_configs
```

The analyzer uses `mean(acquisition, transfer, boundedness, preservation)` as
utility by default and handles oracle winners as sets, so tied top
configurations are all counted as oracle-optimal. Fixed global and
objective-conditioned baselines are learned from TRAIN records only, then
evaluated on validation/test records by default. For legacy records without
preservation, pass `--utility_columns acquisition transfer boundedness`.

## Extract Features

```bash
python src/extract_episode_features.py \
  --episode_manifest data/compiler/episode_manifest.jsonl \
  --examples_path data/prompt_examples.jsonl \
  --model_name_or_path meta-llama/Llama-3.1-8B-Instruct \
  --meta_split train \
  --output_root outputs/compiler/features \
  --backend frozen_model \
  --torch_dtype bfloat16 \
  --device_map auto
```

The default `frozen_model` backend interrogates the frozen backbone before
adaptation. It computes target-token negative log-likelihood on the episode's
adaptation examples, then estimates module sensitivity from gradients with
respect to module outputs. It processes probed modules in small batches; the
default `--probe_module_batch_size 1` is slower but avoids full-backbone
parameter gradients and keeps each probe local.

Gradient agreement is a compact sketch-based approximation over module-output
gradients, not exact parameter-gradient cosine. Use `--max_probe_examples` to
control cost; the target-token loss is still computed from the actual model.
When extracting features for many episodes, pass them in one command with
`--episode_ids` or `--meta_split`; the CLI loads the tokenizer/model once and
reuses the frozen backend across all pending episode outputs.

For unit tests or plumbing-only dry runs, use:

```bash
python src/extract_episode_features.py \
  --episode_manifest data/compiler/episode_manifest.jsonl \
  --examples_path data/prompt_examples.jsonl \
  --model_name_or_path toy/model \
  --backend proxy \
  --max_episodes 1
```

Objective identity is metadata only. It must not enter `episode_features` or
the geometry predictor input vector.

## Train and Evaluate Predictor

```bash
python src/train_geometry_predictor.py \
  --records outputs/compiler/geometry_records.csv \
  --features_root outputs/compiler/features \
  --config_library data/compiler/config_library.jsonl \
  --output_dir outputs/compiler/predictor \
  --seed 2026 \
  --include_preservation
```

The predictor is a small MLP over interpretable episode/configuration features.
It standardizes continuous inputs using train-split statistics only and reports
MAE, per-episode Spearman ranking, and pairwise ranking accuracy. Baselines
include global mean and configuration-only mean outcomes.

Evaluate compiler regret from test predictions:

```bash
python src/evaluate_compiler.py \
  --input outputs/compiler/predictor/predictions_test.csv \
  --train_records outputs/compiler/geometry_records.csv \
  --output_csv outputs/compiler/compiler_evaluation.csv \
  --output_json outputs/compiler/compiler_evaluation_summary.json \
  --utility_weights acquisition=1 transfer=1 boundedness=1 preservation=1
```

The utility function is deliberately configurable. 
