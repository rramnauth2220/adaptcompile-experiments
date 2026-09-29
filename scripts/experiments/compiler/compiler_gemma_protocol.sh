#!/usr/bin/env bash
# Frozen common protocol scaffold for the Gemma-2-9B adaptation-compiler
# replication. This is intentionally separate from scripts/experiments/compiler/compiler_protocol.sh:
# Llama GA=2 remains frozen, while Gemma GA is selected by the Gemma calibration
# sweep and recorded in artifacts/gemma/calibration/selected_schedule.json.

MODEL="google/gemma-2-9b-it"
MANIFEST="data/compiler/episode_manifest.jsonl"
CONFIG_LIBRARY="data/compiler_gemma/config_library.jsonl"
EXAMPLES="data/prompt_examples.jsonl"

PRESERVATION="data/compiler/preservation_examples.jsonl"
PRESERVATION_CACHE="outputs/compiler_gemma/preservation/gemma_2_9b_it_baseline.jsonl"

CONFIGS=(
  early__all__r16
  middle__all__r16
  late__all__r16
  full__all__r4
)

CALIBRATION_CONFIGS=(
  full__all__r16
)

COMMON_ARGS=(
  --model_name_or_path "$MODEL"
  --episode_manifest "$MANIFEST"
  --config_library "$CONFIG_LIBRARY"
  --examples_path "$EXAMPLES"
  --config_ids "${CONFIGS[@]}"
  --seeds 11
  --num_train_epochs 3
  --learning_rate 2e-4
  --per_device_train_batch_size 1
  --max_length 512
  --max_new_tokens 32
  --torch_dtype bfloat16
  --device_map auto
  --preservation_examples_path "$PRESERVATION"
  --preservation_baseline_cache "$PRESERVATION_CACHE"
  --preservation_max_new_tokens 16
  --save_strategy no
  --skip_existing
  --cleanup_adapter_after_eval
)
