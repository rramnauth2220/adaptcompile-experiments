#!/usr/bin/env bash
# Frozen common protocol for calibrated compiler geometry experiments.
# GA=2 was selected by meta-training calibration; this protocol is frozen for
# subsequent geometry experiments.

MODEL="meta-llama/Llama-3.1-8B-Instruct"
MANIFEST="data/compiler/episode_manifest.jsonl"
CONFIG_LIBRARY="data/compiler/config_library.jsonl"
EXAMPLES="data/prompt_examples.jsonl"

PRESERVATION="data/compiler/preservation_examples.jsonl"
PRESERVATION_CACHE="outputs/compiler/preservation/llama31_8b_baseline.jsonl"

CONFIGS=(
  early__all__r16
  middle__all__r16
  late__all__r16
  full__all__r4
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
  --gradient_accumulation_steps 2
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
