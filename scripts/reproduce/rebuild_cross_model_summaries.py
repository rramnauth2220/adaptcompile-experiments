#!/usr/bin/env python3

from pathlib import Path
import argparse
import pandas as pd
import re


MODEL_NAME_BY_SLUG = {
    "llama_3_1_8b_instruct": "meta-llama/Llama-3.1-8B-Instruct",
    "mistral_7b_instruct_v0_3": "mistralai/Mistral-7B-Instruct-v0.3",
    "gemma_2_9b_it": "google/gemma-2-9b-it",
    "olmo_2_1124_7b_instruct": "allenai/OLMo-2-1124-7B-Instruct",
    "qwen2_5_14b_instruct": "Qwen/Qwen2.5-14B-Instruct",
}


def parse_path(path: Path):
    parts = path.parts
    # Expected:
    # outputs/cross_model_localization/<model_slug>/<objective>/budget_<B>/condition_<cond>/rank_<R>/seed_<S>/summary_localization_by_seed.csv
    idx = parts.index("cross_model_localization")
    model_slug = parts[idx + 1]
    objective = parts[idx + 2]
    budget = int(parts[idx + 3].replace("budget_", ""))
    condition = parts[idx + 4].replace("condition_", "")
    rank = int(parts[idx + 5].replace("rank_", ""))
    seed = int(parts[idx + 6].replace("seed_", ""))
    output_dir = Path(*parts[: idx + 7])
    return model_slug, objective, budget, condition, rank, seed, output_dir


def pick_metric(row, candidates):
    for c in candidates:
        if c in row.index:
            return row[c]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs/cross_model_localization")
    args = ap.parse_args()

    root = Path(args.root)
    summary_files = sorted(root.glob("*/*/budget_*/condition_*/rank_*/seed_*/summary_localization_by_seed.csv"))

    rows = []
    manifest_rows = []

    for sf in summary_files:
        try:
            model_slug, objective, budget, condition, rank, seed, output_dir = parse_path(sf)
        except Exception as e:
            print(f"[WARN] Could not parse path {sf}: {e}")
            continue

        df = pd.read_csv(sf)
        if len(df) == 0:
            print(f"[WARN] Empty summary file: {sf}")
            continue

        # Most per-run summaries should have one useful row. If multiple, use the first.
        r = df.iloc[0]

        id_eval = pick_metric(r, ["id_eval", "id", "id_accuracy", "id_strict_accuracy"])
        paraphrase_eval = pick_metric(r, ["paraphrase_eval", "paraphrase", "paraphrase_accuracy", "paraphrase_strict_accuracy"])
        acquisition = pick_metric(r, ["acquisition"])
        transfer = pick_metric(r, ["transfer", "generalization"])
        generalization = pick_metric(r, ["generalization", "transfer"])
        boundedness = pick_metric(r, ["boundedness", "negative_control", "neg", "neg_ctrl"])

        # If acquisition was not precomputed, compute it from ID + paraphrase.
        if acquisition is None and id_eval is not None and paraphrase_eval is not None:
            acquisition = (float(id_eval) + float(paraphrase_eval)) / 2.0

        model_name = MODEL_NAME_BY_SLUG.get(model_slug, model_slug)

        rows.append({
            "model_name": model_name,
            "model_slug": model_slug,
            "objective": objective,
            "budget": budget,
            "condition": condition,
            "seed": seed,
            "rank": rank,
            "id_eval": id_eval,
            "paraphrase_eval": paraphrase_eval,
            "acquisition": acquisition,
            "transfer": transfer,
            "generalization": generalization,
            "boundedness": boundedness,
            "output_dir": str(output_dir),
            "summary_file": str(sf),
        })

        result_jsonls = list(output_dir.glob("*.jsonl")) + list(output_dir.glob("*.jsonl.gz"))
        result_jsonl = str(result_jsonls[0]) if result_jsonls else ""

        manifest_rows.append({
            "model_name": model_name,
            "model_slug": model_slug,
            "objective": objective,
            "budget": budget,
            "condition": condition,
            "seed": seed,
            "rank": rank,
            "output_dir": str(output_dir),
            "adapter_dir": str(output_dir / "adapter"),  # may have been deleted; kept as historical path
            "result_jsonl": result_jsonl,
            "per_run_summary_csv": str(sf),
            "run_id": f"crossmodel::{model_slug}::{objective}::budget{budget}::{condition}::rank{rank}::seed{seed}",
            "command": "",
        })

    by_seed = pd.DataFrame(rows)
    if by_seed.empty:
        raise SystemExit(f"No summary files found under {root}")

    by_seed = by_seed.sort_values(["model_slug", "objective", "condition", "seed"])

    agg = (
        by_seed
        .groupby(["model_name", "model_slug", "objective", "budget", "condition", "rank"], dropna=False)
        .agg(
            acquisition_mean=("acquisition", "mean"),
            acquisition_sd=("acquisition", "std"),
            transfer_mean=("transfer", "mean"),
            transfer_sd=("transfer", "std"),
            generalization_mean=("generalization", "mean"),
            generalization_sd=("generalization", "std"),
            boundedness_mean=("boundedness", "mean"),
            boundedness_sd=("boundedness", "std"),
            n_seeds=("seed", "nunique"),
        )
        .reset_index()
    )

    manifest = pd.DataFrame(manifest_rows).sort_values(["model_slug", "objective", "condition", "seed"])

    by_seed.to_csv(root / "summary_cross_model_by_seed.csv", index=False)
    agg.to_csv(root / "summary_cross_model_aggregate.csv", index=False)
    manifest.to_csv(root / "cross_model_manifest.csv", index=False)

    print(f"Wrote {root / 'summary_cross_model_by_seed.csv'} rows={len(by_seed)}")
    print(f"Wrote {root / 'summary_cross_model_aggregate.csv'} rows={len(agg)}")
    print(f"Wrote {root / 'cross_model_manifest.csv'} rows={len(manifest)}")

    print("\nCoverage by model/objective:")
    print(by_seed.groupby(["model_slug", "objective"])["seed"].nunique().unstack(fill_value=0))


if __name__ == "__main__":
    main()
