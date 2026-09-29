# Running Calibration Experiments

Calibration asks a simple question before localized adaptation is interpreted:
can the model learn this objective at all under ordinary full-stack LoRA, and
which training budget gives a useful, non-saturated regime?

The basic loop is:

1. Evaluate the base model before adaptation.
2. Train a full-stack LoRA adapter for one manifest `run_id`.
3. Evaluate the adapted model on the same evaluation splits.
4. Summarize raw JSONL outputs into CSV rows.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Choose a Run ID

Run IDs live in `data/calibration_manifest.jsonl` and have the form:

```text
calib::<learning_type>::specs<n_specs>::budget<budget>
```

Examples:

```text
calib::lexical_binding::specs25::budget10
calib::factual_association::specs25::budget8
calib::behavioral_policy::specs25::budget10
calib::causal_mapping::specs25::budget10
calib::procedural_reasoning::specs25::budget8
```

Check that a planned run exists:

```bash
python scripts/maintenance/check_calibration_manifest_runids.py \
  --learning_type lexical_binding \
  --n_specs 25 \
  --budgets 10
```

## Baseline Evaluation

Use the objective-specific evaluator when one exists. The generic evaluator
handles lexical and factual runs.

```bash
python src/evaluate_calibration_run.py \
  --model_name Qwen/Qwen2.5-0.5B-Instruct \
  --run_id 'calib::lexical_binding::specs25::budget10' \
  --output_path outputs/baseline_calib_lexical_binding_specs25_budget10.jsonl \
  --max_new_tokens 32
```

Specialized evaluators:

- `src/evaluate_calibration_run_behavioral.py`
- `src/evaluate_calibration_run_causal.py`
- `src/evaluate_calibration_run_procedural.py`

They use the same main arguments: `--model_name`, `--adapter_path`,
`--run_id`, `--output_path`, and `--max_new_tokens`.

## Full-Stack LoRA Training

```bash
python src/train_fullstack_lora.py \
  --model_name Qwen/Qwen2.5-0.5B-Instruct \
  --run_id 'calib::lexical_binding::specs25::budget10' \
  --output_dir outputs/calib_lexical_binding_specs25_budget10 \
  --target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
  --num_train_epochs 3 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8
```

For GPT-2-like models, use:

```bash
--target_modules c_attn c_proj
```

Use `--seed` for reproducible multi-seed sweeps.

## Adapter Evaluation

```bash
python src/evaluate_calibration_run.py \
  --model_name Qwen/Qwen2.5-0.5B-Instruct \
  --adapter_path outputs/calib_lexical_binding_specs25_budget10 \
  --run_id 'calib::lexical_binding::specs25::budget10' \
  --output_path outputs/adapter_calib_lexical_binding_specs25_budget10.jsonl \
  --max_new_tokens 32
```

## Summarize

```bash
python src/summarize_results.py \
  --inputs \
    outputs/baseline_calib_lexical_binding_specs25_budget10.jsonl \
    outputs/adapter_calib_lexical_binding_specs25_budget10.jsonl \
  --output_csv outputs/summary_calib_lexical_binding_specs25_budget10.csv
```

To also make appendix-style figures:

```bash
python src/summarize_results.py \
  --inputs outputs/llama31_8b_lexical_coarse/*.jsonl.gz \
  --output_csv outputs/llama31_8b_lexical_coarse/summary_lexical_binding_specs25_all_budgets.csv \
  --figure_dir outputs/llama31_8b_lexical_coarse/figures
```

## One-Command Smoke Run

```bash
bash scripts/experiments/calibration/run_smoke_calibration.sh
```

Environment variables can override defaults:

```bash
MODEL=Qwen/Qwen2.5-0.5B-Instruct \
RUN_ID='calib::lexical_binding::specs25::budget8' \
OUT_DIR=outputs/smoke_lexical \
RESULTS_DIR=outputs/smoke_lexical \
bash scripts/experiments/calibration/run_smoke_calibration.sh
```

## Budget Sweep

For coarse calibration across budgets:

```bash
LEARNING_TYPE=lexical_binding \
N_SPECS=25 \
BUDGETS='1 2 4 8 12' \
bash scripts/experiments/calibration/run_mode_aware_budget_sweep.sh
```

The script writes adapters to `outputs/mode_aware_budget_sweep/` and raw
evaluation/summaries to `outputs/mode_aware_budget_sweep/` unless overridden.

## Interpreting Calibration

A useful calibration regime usually has:

- low pre-adaptation baseline performance;
- high but not totally saturated ID/paraphrase acquisition after adaptation;
- nontrivial generalization;
- low negative-control false positives;
- comparable learnability across objectives before localized adaptation is
  compared.

The released localization evidence in `artifacts/localization/primary/` uses
calibrated budgets of:

- lexical binding: `10`
- factual association: `8`
- behavioral policy: `10`
- causal mapping: `10`
- procedural reasoning: `8`
