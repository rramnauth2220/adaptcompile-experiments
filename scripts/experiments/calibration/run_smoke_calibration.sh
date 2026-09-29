#!/usr/bin/env bash
set -euo pipefail
MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
RUN_ID="${RUN_ID:-calib::lexical_binding::specs25::budget8}"
OUT_DIR="${OUT_DIR:-outputs/calib_lexical_specs25_budget8}"
RESULTS_DIR="${RESULTS_DIR:-results}"
SAFE_RUN_ID="${RUN_ID//::/_}"
mkdir -p "$RESULTS_DIR"

python src/evaluate_calibration_run.py \
  --model_name "$MODEL" \
  --run_id "$RUN_ID" \
  --output_path "$RESULTS_DIR/baseline_${SAFE_RUN_ID}.jsonl" \
  --max_new_tokens 32

python src/train_fullstack_lora.py \
  --model_name "$MODEL" \
  --run_id "$RUN_ID" \
  --output_dir "$OUT_DIR" \
  --target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
  --num_train_epochs 3 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8

python src/evaluate_calibration_run.py \
  --model_name "$MODEL" \
  --adapter_path "$OUT_DIR" \
  --run_id "$RUN_ID" \
  --output_path "$RESULTS_DIR/adapter_${SAFE_RUN_ID}.jsonl" \
  --max_new_tokens 32

python src/summarize_results.py \
  --inputs "$RESULTS_DIR/baseline_${SAFE_RUN_ID}.jsonl" "$RESULTS_DIR/adapter_${SAFE_RUN_ID}.jsonl" \
  --output_csv "$RESULTS_DIR/summary_${SAFE_RUN_ID}.csv"
