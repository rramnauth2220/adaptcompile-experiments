#!/usr/bin/env python3
"""Evaluate a base model or a PEFT adapter on generated examples."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from common import read_jsonl, write_jsonl, format_prompt, score_response

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", required=True)
    p.add_argument("--adapter_path", default=None)
    p.add_argument("--examples_path", default="data/prompt_examples.jsonl")
    p.add_argument("--output_path", required=True)
    p.add_argument("--splits", nargs="+", default=["id_eval", "paraphrase_eval", "generalization", "negative_control"])
    p.add_argument("--learning_types", nargs="+", default=None)
    p.add_argument("--max_examples_per_split", type=int, default=None)
    p.add_argument("--max_new_tokens", type=int, default=32)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--device_map", default="auto")
    p.add_argument("--torch_dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    return p.parse_args()

def dtype_from_arg(arg):
    if arg == "auto": return "auto"
    return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[arg]

def main():
    args = parse_args()
    examples = read_jsonl(args.examples_path)
    examples = [e for e in examples if e["split"] in set(args.splits)]
    if args.learning_types:
        examples = [e for e in examples if e["learning_type"] in set(args.learning_types)]

    if args.max_examples_per_split is not None:
        kept, counts = [], Counter()
        for e in examples:
            key = (e["learning_type"], e["split"])
            if counts[key] < args.max_examples_per_split:
                kept.append(e); counts[key] += 1
        examples = kept

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model_name, device_map=args.device_map, torch_dtype=dtype_from_arg(args.torch_dtype))
    if args.adapter_path:
        model = PeftModel.from_pretrained(model, args.adapter_path)
    model.eval()

    rows = []
    for e in tqdm(examples, desc="evaluating"):
        prompt = format_prompt(e["prompt"])
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.temperature > 0,
                temperature=args.temperature if args.temperature > 0 else None,
                pad_token_id=tokenizer.eos_token_id,
            )
        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        score = score_response(e, response)
        rows.append({
            "example_id": e["example_id"],
            "spec_id": e["spec_id"],
            "learning_type": e["learning_type"],
            "split": e["split"],
            "prompt": e["prompt"],
            "target": e["target"],
            "response": response,
            **score,
        })

    write_jsonl(args.output_path, rows)
    summary = defaultdict(lambda: Counter())
    for r in rows:
        summary[(r["learning_type"], r["split"])]["n"] += 1
        summary[(r["learning_type"], r["split"])]["passed"] += int(r["passed"])
    print("\nSummary")
    print("-------")
    for key in sorted(summary):
        n = summary[key]["n"]; passed = summary[key]["passed"]
        print(f"{key[0]:22s} {key[1]:18s}: {passed:4d}/{n:<4d} = {passed/max(n,1):.3f}")

if __name__ == "__main__":
    main()
