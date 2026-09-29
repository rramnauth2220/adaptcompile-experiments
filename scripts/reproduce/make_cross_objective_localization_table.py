#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


OBJECTIVES = [
    {
        "key": "lexical_binding",
        "label": "Lexical binding",
        "budget": 10,
        "match": "lexical",
        "boundedness_source": "strict",
    },
    {
        "key": "factual_association",
        "label": "Factual association",
        "budget": 8,
        "match": "factual",
        "boundedness_source": "strict",
    },
    {
        "key": "behavioral_policy",
        "label": "Behavioral policy",
        "budget": 10,
        "match": "behavioral",
        "boundedness_source": "concept",
    },
    {
        "key": "causal_mapping",
        "label": "Causal mapping",
        "budget": 10,
        "match": "causal",
        "boundedness_source": "concept",
    },
    {
        "key": "procedural_reasoning",
        "label": "Procedural reasoning",
        "budget": 8,
        "match": "procedural",
        "boundedness_source": "concept",
    },
]

CONDITION_ORDER = ["full", "early", "middle", "late"]
CONDITION_LABELS = {
    "full": "Full",
    "early": "Early",
    "middle": "Middle",
    "late": "Late",
}


def load_release_by_seed(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    out = pd.read_csv(path)
    required = {
        "seed", "localization_condition", "id_eval", "paraphrase_eval",
        "acquisition", "generalization", "boundedness", "objective_key",
        "objective", "calibrated_budget", "source_dir",
    }
    missing = required - set(out.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    out["localization_condition"] = out["localization_condition"].str.lower()
    return out[out["localization_condition"].isin(CONDITION_ORDER)].copy()


def mean_sd(series: pd.Series) -> str:
    mean = series.mean()
    sd = series.std()
    return f"{mean:.3f} $\\pm$ {sd:.3f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate the cross-objective localization table from compact release evidence.")
    parser.add_argument("--input", default="artifacts/localization/primary/cross_objective_localization/cross_objective_localization_table_by_seed.csv")
    parser.add_argument("--output-dir", default="outputs/release_verification/tables/cross_objective_localization")
    args = parser.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    per_seed = load_release_by_seed(Path(args.input))

    objective_order = [o["key"] for o in OBJECTIVES]
    per_seed["objective_key"] = pd.Categorical(
        per_seed["objective_key"], objective_order, ordered=True
    )
    per_seed["localization_condition"] = pd.Categorical(
        per_seed["localization_condition"], CONDITION_ORDER, ordered=True
    )
    per_seed = per_seed.sort_values(
        ["objective_key", "localization_condition", "seed"]
    )

    per_seed_path = outdir / "cross_objective_localization_table_by_seed.csv"
    per_seed.to_csv(per_seed_path, index=False)

    agg = (
        per_seed.groupby(
            [
                "objective_key",
                "objective",
                "calibrated_budget",
                "localization_condition",
            ],
            observed=True,
            as_index=False,
        )
        .agg(
            id_eval_mean=("id_eval", "mean"),
            id_eval_sd=("id_eval", "std"),
            paraphrase_mean=("paraphrase_eval", "mean"),
            paraphrase_sd=("paraphrase_eval", "std"),
            acquisition_mean=("acquisition", "mean"),
            acquisition_sd=("acquisition", "std"),
            transfer_mean=("generalization", "mean"),
            transfer_sd=("generalization", "std"),
            boundedness_mean=("boundedness", "mean"),
            boundedness_sd=("boundedness", "std"),
            n_seeds=("seed", "nunique"),
        )
    )

    expected_rows = len(OBJECTIVES) * len(CONDITION_ORDER)
    if len(agg) != expected_rows:
        print(agg)
        raise ValueError(f"Expected {expected_rows} rows, found {len(agg)}.")

    raw_path = outdir / "cross_objective_localization_table_raw.csv"
    agg.to_csv(raw_path, index=False)

    formatted_rows = []

    for _, r in agg.iterrows():
        condition = str(r["localization_condition"])
        formatted_rows.append(
            {
                "Objective": r["objective"],
                "Calibrated Budget": int(r["calibrated_budget"]),
                "Condition": CONDITION_LABELS[condition],
                "ID": f"{r['id_eval_mean']:.3f} ± {r['id_eval_sd']:.3f}",
                "Paraphrase": f"{r['paraphrase_mean']:.3f} ± {r['paraphrase_sd']:.3f}",
                "Acquisition": f"{r['acquisition_mean']:.3f} ± {r['acquisition_sd']:.3f}",
                "Transfer": f"{r['transfer_mean']:.3f} ± {r['transfer_sd']:.3f}",
                "Boundedness": f"{r['boundedness_mean']:.3f} ± {r['boundedness_sd']:.3f}",
                "N seeds": int(r["n_seeds"]),
            }
        )

    formatted = pd.DataFrame(formatted_rows)

    formatted_path = outdir / "cross_objective_localization_table_formatted.csv"
    formatted.to_csv(formatted_path, index=False)

    # LaTeX table.
    latex = r"""\begin{table*}[t]
\centering
\small
\begin{tabular}{llcccccc}
\toprule
Objective & Condition & Budget & ID & Para & Acquisition & Transfer & Boundedness \\
\midrule
"""

    def tex_pm(value: str) -> str:
        return str(value).replace("±", r"$\pm$")


    current_obj = None
    for _, r in formatted.iterrows():
        obj = r["Objective"]

        if current_obj is not None and obj != current_obj:
            latex += r"\midrule" + "\n"
        current_obj = obj

        id_tex = tex_pm(r["ID"])
        para_tex = tex_pm(r["Paraphrase"])
        acq_tex = tex_pm(r["Acquisition"])
        transfer_tex = tex_pm(r["Transfer"])
        bounded_tex = tex_pm(r["Boundedness"])

        latex += (
            f"{obj} & {r['Condition']} & {r['Calibrated Budget']} & "
            f"{id_tex} & {para_tex} & {acq_tex} & "
            f"{transfer_tex} & {bounded_tex} \\\\\n"
        )

    latex += r"""\bottomrule
\end{tabular}
\caption{
Cross-objective localization summary. Each objective is evaluated at its selected calibration budget. ID and Para report in-distribution and paraphrase accuracy, respectively. Acquisition is the mean of ID and Para. Transfer is accuracy on held-out generalization examples. Boundedness is objective-specific negative-control accuracy. Values are mean $\pm$ standard deviation across seeds.
}
\label{tab:cross-objective-localization}
\end{table*}
"""

    latex_path = outdir / "cross_objective_localization_table.tex"
    latex_path.write_text(latex, encoding="utf-8")

    print("Wrote:")
    print(per_seed_path)
    print(raw_path)
    print(formatted_path)
    print(latex_path)

    print("\nPreview:")
    print(formatted.to_string(index=False))

    print("\nSources:")
    for src in per_seed["source_dir"].drop_duplicates():
        print(src)


if __name__ == "__main__":
    main()
