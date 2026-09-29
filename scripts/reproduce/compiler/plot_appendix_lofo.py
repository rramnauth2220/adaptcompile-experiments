#!/usr/bin/env python3
"""Appendix Figure D: leave-one-objective-out generalization."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from plot_style import (
    OBJECTIVE_LABELS,
    OBJECTIVE_ORDER,
    SELECTOR_COLORS,
    SELECTOR_LABELS,
    SELECTOR_MARKERS,
    ensure_dirs,
    format_number,
    json_script_metadata,
    panel_label,
    prettify_axes,
    read_csv,
    save_outputs,
    setup_matplotlib,
    write_summary,
)


def build_rows(lofo_summary_path: Path):
    df = read_csv(lofo_summary_path)
    expected = set(OBJECTIVE_ORDER + ["macro_average"])
    observed = set(df["held_out_learning_type"].astype(str))
    missing = expected - observed
    if missing:
        raise ValueError(f"LOFO summary missing expected rows: {sorted(missing)}")
    order = OBJECTIVE_ORDER + ["macro_average"]
    df["_order"] = df["held_out_learning_type"].apply(lambda x: order.index(str(x)))
    df = df.sort_values("_order")
    rows = []
    for _, row in df.iterrows():
        lt = str(row["held_out_learning_type"])
        label = OBJECTIVE_LABELS.get(lt, "Macro")
        rows.append(
            {
                "held_out_learning_type": lt,
                "objective": label,
                "n_test_episodes": int(row["n_test_episodes"]),
                "global_fixed_mean_utility": float(row["global_fixed_mean_utility"]),
                "compiler_mean_utility": float(row["compiler_mean_utility"]),
                "oracle_mean_utility": float(row["oracle_mean_utility"]),
                "global_fixed_oracle_regret": float(row["global_fixed_oracle_regret"]),
                "compiler_oracle_regret": float(row["compiler_oracle_regret"]),
                "compiler_oracle_optimal_fraction": float(row["compiler_oracle_optimal_fraction"]),
                "global_fixed_oracle_optimal_fraction": float(row["global_fixed_oracle_optimal_fraction"]),
            }
        )
    return rows


def draw_figure(rows, output_root: Path, inputs):
    plt = setup_matplotlib()
    fig, (ax_a, ax_b) = plt.subplots(2, 1, figsize=(6.75, 4.55), sharex=True)
    x = np.arange(len(rows))
    offsets = {"global_fixed": -0.18, "lofo_compiler": 0.0, "oracle": 0.18}
    utility_cols = {
        "global_fixed": "global_fixed_mean_utility",
        "lofo_compiler": "compiler_mean_utility",
        "oracle": "oracle_mean_utility",
    }
    for selector in ["global_fixed", "lofo_compiler", "oracle"]:
        ax_a.scatter(
            x + offsets[selector],
            [row[utility_cols[selector]] for row in rows],
            marker=SELECTOR_MARKERS[selector],
            color=SELECTOR_COLORS[selector],
            s=38 if selector != "oracle" else 55,
            label=SELECTOR_LABELS[selector],
            zorder=4,
        )
    ax_a.set_ylabel("Realized utility")
    ax_a.set_title("Held-out-objective utility", loc="left")
    ax_a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, frameon=False)
    prettify_axes(ax_a)
    panel_label(ax_a, "A")

    regret_cols = {
        "global_fixed": "global_fixed_oracle_regret",
        "lofo_compiler": "compiler_oracle_regret",
    }
    for selector in ["global_fixed", "lofo_compiler"]:
        ax_b.scatter(
            x + offsets[selector],
            [row[regret_cols[selector]] for row in rows],
            marker=SELECTOR_MARKERS[selector],
            color=SELECTOR_COLORS[selector],
            s=38,
            zorder=4,
            label=SELECTOR_LABELS[selector],
        )
        ax_b.plot(x + offsets[selector], [row[regret_cols[selector]] for row in rows], color=SELECTOR_COLORS[selector], linewidth=0.8, alpha=0.65)
    ax_b.axhline(0, color="#777777", linewidth=0.8)
    ax_b.set_ylabel("Oracle regret")
    ax_b.set_title("Held-out-objective regret", loc="left")
    ax_b.set_xticks(x, [row["objective"] for row in rows])
    prettify_axes(ax_b)
    panel_label(ax_b, "B")
    fig.tight_layout(h_pad=2.0)

    dirs = ensure_dirs(output_root)
    write_summary(dirs["data"] / "lofo_summary_plot.csv", rows)
    metadata = json_script_metadata(__file__, inputs)
    metadata.update({"summary": rows, "source_guard": "LOFO figure uses lofo_summary.csv only."})
    save_outputs(fig, dirs["appendix"] / "fig_lofo_generalization", metadata)
    plt.close(fig)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot appendix LOFO generalization figure.")
    parser.add_argument("--lofo_summary", default="artifacts/llama/lofo/lofo_summary.csv")
    parser.add_argument("--output_root", default="outputs/release_verification/figures/llama")
    args = parser.parse_args()
    lofo = Path(args.lofo_summary)
    rows = build_rows(lofo)
    metadata = draw_figure(rows, Path(args.output_root), [lofo])
    macro = next(row for row in metadata["summary"] if row["held_out_learning_type"] == "macro_average")
    print(
        "Appendix LOFO:",
        f"macro_compiler_regret={format_number(macro['compiler_oracle_regret'])}",
        f"macro_global_regret={format_number(macro['global_fixed_oracle_regret'])}",
    )


if __name__ == "__main__":
    main()
