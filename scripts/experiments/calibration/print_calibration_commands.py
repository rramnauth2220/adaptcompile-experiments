#!/usr/bin/env python3
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
RUN_IDS = [
    "calib::lexical_binding::specs10::budget1",
    "calib::lexical_binding::specs10::budget2",
    "calib::lexical_binding::specs10::budget4",
    "calib::factual_association::specs10::budget1",
    "calib::factual_association::specs10::budget2",
    "calib::factual_association::specs10::budget4",
]
for run_id in RUN_IDS:
    safe = run_id.replace("::", "_")
    out_dir = f"outputs/{safe}"
    print(f"python src/evaluate_calibration_run.py --model_name {MODEL} --run_id '{run_id}' --output_path outputs/baseline_{safe}.jsonl --max_new_tokens 32")
    print(f"python src/train_fullstack_lora.py --model_name {MODEL} --run_id '{run_id}' --output_dir {out_dir} --target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj --num_train_epochs 1 --per_device_train_batch_size 1 --gradient_accumulation_steps 8")
    print(f"python src/evaluate_calibration_run.py --model_name {MODEL} --adapter_path {out_dir} --run_id '{run_id}' --output_path outputs/adapter_{safe}.jsonl --max_new_tokens 32\n")
