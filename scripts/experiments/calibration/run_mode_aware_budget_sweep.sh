#!/usr/bin/env bash
# Run mode-aware full-stack calibration across budgets and save raw JSONL outputs.
#
# Fix included:
#   train_fullstack_lora.py requires --target_modules, so this script passes
#   Qwen/Llama-style defaults.

set -euo pipefail

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-0.5B-Instruct}"
LEARNING_TYPE="${LEARNING_TYPE:-lexical_binding}"
N_SPECS="${N_SPECS:-25}"
BUDGETS="${BUDGETS:-1 2 4 8 12}"
NUM_EPOCHS="${NUM_EPOCHS:-3}"

MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}"
LR="${LR:-2e-4}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"

# For Qwen/Llama/Mistral-style models.
TARGET_MODULES="${TARGET_MODULES:-q_proj k_proj v_proj o_proj gate_proj up_proj down_proj}"

RUN_ROOT="${RUN_ROOT:-outputs/mode_aware_budget_sweep}"
RESULTS_ROOT="${RESULTS_ROOT:-outputs/mode_aware_budget_sweep}"

mkdir -p "${RUN_ROOT}" "${RESULTS_ROOT}"

echo "Mode-aware budget sweep"
echo "-----------------------"
echo "MODEL_NAME=${MODEL_NAME}"
echo "LEARNING_TYPE=${LEARNING_TYPE}"
echo "N_SPECS=${N_SPECS}"
echo "BUDGETS=${BUDGETS}"
echo "NUM_EPOCHS=${NUM_EPOCHS}"
echo "TARGET_MODULES=${TARGET_MODULES}"
echo "RUN_ROOT=${RUN_ROOT}"
echo "RESULTS_ROOT=${RESULTS_ROOT}"
echo

ALL_JSONL=()

for BUDGET in ${BUDGETS}; do
  RUN_ID="calib::${LEARNING_TYPE}::specs${N_SPECS}::budget${BUDGET}"
  SAFE_RUN="calib_${LEARNING_TYPE}_specs${N_SPECS}_budget${BUDGET}"

  OUT_DIR="${RUN_ROOT}/${SAFE_RUN}"
  BASELINE_JSONL="${RESULTS_ROOT}/baseline_${SAFE_RUN}.jsonl"
  ADAPTER_JSONL="${RESULTS_ROOT}/adapter_${SAFE_RUN}.jsonl"
  SUMMARY_CSV="${RESULTS_ROOT}/summary_${SAFE_RUN}.csv"

  echo
  echo "============================================================"
  echo "Running ${RUN_ID}"
  echo "Adapter output: ${OUT_DIR}"
  echo "Baseline raw:   ${BASELINE_JSONL}"
  echo "Adapter raw:    ${ADAPTER_JSONL}"
  echo "Summary:        ${SUMMARY_CSV}"
  echo "============================================================"

  echo "[1/4] Baseline evaluation"
  python src/evaluate_calibration_run.py \
    --model_name "${MODEL_NAME}" \
    --run_id "${RUN_ID}" \
    --output_path "${BASELINE_JSONL}" \
    --max_new_tokens "${MAX_NEW_TOKENS}"

  echo "[2/4] Full-stack LoRA training"
  python src/train_fullstack_lora.py \
    --model_name "${MODEL_NAME}" \
    --run_id "${RUN_ID}" \
    --output_dir "${OUT_DIR}" \
    --target_modules ${TARGET_MODULES} \
    --num_train_epochs "${NUM_EPOCHS}" \
    --learning_rate "${LR}" \
    --lora_r "${LORA_R}" \
    --lora_alpha "${LORA_ALPHA}" \
    --lora_dropout "${LORA_DROPOUT}" \
    --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}"

  echo "[3/4] Adapter evaluation"
  python src/evaluate_calibration_run.py \
    --model_name "${MODEL_NAME}" \
    --adapter_path "${OUT_DIR}" \
    --run_id "${RUN_ID}" \
    --output_path "${ADAPTER_JSONL}" \
    --max_new_tokens "${MAX_NEW_TOKENS}"

  echo "[4/4] Summary"
  python src/summarize_results.py \
    --inputs "${BASELINE_JSONL}" "${ADAPTER_JSONL}" \
    --output_csv "${SUMMARY_CSV}"

  ALL_JSONL+=("${BASELINE_JSONL}" "${ADAPTER_JSONL}")
done

COMBINED_SUMMARY="${RESULTS_ROOT}/summary_${LEARNING_TYPE}_specs${N_SPECS}_all_budgets.csv"

echo
echo "Writing combined summary: ${COMBINED_SUMMARY}"
python src/summarize_results.py \
  --inputs "${ALL_JSONL[@]}" \
  --output_csv "${COMBINED_SUMMARY}"

echo
echo "Done."
echo "Raw JSONL files are in: ${RESULTS_ROOT}"
echo "Adapters are in: ${RUN_ROOT}"
echo "Combined summary: ${COMBINED_SUMMARY}"
