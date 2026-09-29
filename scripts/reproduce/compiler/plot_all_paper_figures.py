#!/usr/bin/env python3
"""Regenerate all paper-facing compiler figures from saved results."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


def run_command(cmd: List[str]) -> None:
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def figure_manifest(output_root: Path) -> str:
    rows = [
        {
            "filename": "main/fig_exp1_headroom.pdf, main/fig_exp1_headroom.png",
            "placement": "Main",
            "question": "Do different episodes prefer different adaptation programs?",
            "panels": "A: tie-aware oracle program shares by objective. B: objective-fixed regret/headroom by episode on the same objective axis.",
            "sources": "artifacts/llama/geometry/geometry_dataset.csv",
            "takeaway": (
                "Oracle-optimal programs vary across learning objectives, and many held-out episodes retain "
                "measurable episode-specific headroom beyond objective-fixed selection."
            ),
            "caption": (
                "Experiment 1 shows selection headroom in the candidate adaptation-program library. "
                "Panel A reports tie-aware oracle-optimal program shares by learning objective; Panel B "
                "shows episode-level utility gained by oracle selection relative to a train-derived "
                "objective-fixed baseline on the same objective axis."
            ),
        },
        {
            "filename": "main/fig_exp2_prediction.pdf, main/fig_exp2_prediction.png",
            "placement": "Main",
            "question": "Can pre-adaptation signals predict adaptation geometry?",
            "panels": "A: predicted versus observed utility. B: geometry prediction error. C: within-episode program ranking.",
            "sources": "artifacts/llama/prediction/*",
            "takeaway": (
                "The primary predictor tracks observed utility, improves outcome-level geometry prediction, "
                "and better preserves within-episode program order."
            ),
            "caption": (
                "Predicting adaptation geometry from pre-adaptation episode--model signals. (A) Predicted and "
                "observed balanced utility across held-out episode--program pairs; the dashed line indicates "
                "perfect agreement. (B) Mean absolute error for acquisition, transfer, boundedness, and "
                "preservation under the primary geometry predictor and non-episode-conditioned baselines. "
                "(C) Within-episode program ranking measured by Spearman correlation, pairwise ranking accuracy, "
                "and tie-aware top-1 and top-2 oracle recovery."
            ),
        },
        {
            "filename": "appendix/fig_exp2_by_objective.pdf, appendix/fig_exp2_by_objective.png",
            "placement": "Appendix",
            "question": "Is Experiment 2 prediction performance distributed across learning objectives?",
            "panels": "A: episode-level geometry prediction error by objective. B: within-episode program-ranking fidelity by objective.",
            "sources": (
                "artifacts/llama/prediction/predictions_test.csv; "
                "artifacts/llama/prediction/ranking_metrics_by_episode.csv; "
                "artifacts/llama/prediction/ranking_metrics.json; "
                "artifacts/llama/prediction/model_metadata.json"
            ),
            "takeaway": (
                "The primary geometry predictor remains informative across represented learning objectives while "
                "exposing objective-level variation in prediction error and ranking fidelity."
            ),
            "caption": (
                "Geometry prediction across learning objectives. (A) Episode-level mean absolute error between "
                "predicted and observed adaptation geometry, grouped by learning objective. Points denote held-out "
                "episodes and summary markers show means with 95% bootstrap confidence intervals. (B) Within-episode "
                "agreement between predicted and observed program orderings, measured by Spearman rank correlation "
                "and pairwise ranking accuracy. The same primary predictor is used for all objectives and is not "
                "given objective identity."
            ),
        },
        {
            "filename": "main/fig_exp3_selection.pdf, main/fig_exp3_selection.png",
            "placement": "Main",
            "question": "Does predicted geometry improve program selection?",
            "panels": "A: realized utility for fixed baselines, compiler, and oracle. B: regret under utility specifications.",
            "sources": "artifacts/llama/geometry/geometry_dataset.csv; artifacts/llama/selection/*",
            "takeaway": "The compiler reduces regret relative to fixed baselines and approaches the oracle under several utility specifications.",
            "caption": (
                "Experiment 3 turns predicted geometry into a single executable adaptation-program choice. "
                "The compiler reduces oracle regret relative to global and objective-fixed baselines, while the "
                "oracle marks the exhaustive-search ceiling."
            ),
        },
        {
            "filename": "main/fig_exp4_ablation.pdf, main/fig_exp4_ablation.png",
            "placement": "Main",
            "question": "Which pre-adaptation representation is sufficient for compilation?",
            "panels": "A: geometry MAE. B: compiler oracle regret. C: tie-aware top-1 oracle recovery.",
            "sources": "artifacts/llama/ablations/*; artifacts/llama/prediction",
            "takeaway": (
                "The episode-only representation retains performance comparable to the full representation, "
                "while module-probe and frozen-behavior features are weaker when used independently."
            ),
            "caption": (
                "Representation ablations. Geometry prediction and downstream selection are compared across "
                "the full episode--model representation and three restricted feature sets. (A) Mean "
                "adaptation-geometry prediction error. (B) Oracle regret of the program selected from "
                "predicted geometry. (C) Tie-aware top-1 recovery of an oracle-optimal program. The episode "
                "representation alone retains performance comparable to the full representation, while "
                "module-level probes and frozen-model behavioral statistics are weaker when used independently."
            ),
        },
        {
            "filename": "main/fig_exp5_generalization.pdf, main/fig_exp5_generalization.png",
            "placement": "Main",
            "question": "Does the adaptation compiler generalize beyond represented learning families?",
            "panels": "A: represented-family versus LOFO geometry prediction error. B: LOFO regret reduction relative to the global-fixed policy.",
            "sources": (
                "artifacts/llama/prediction/predictions_test.csv; "
                "artifacts/llama/prediction/model_metadata.json; "
                "artifacts/llama/lofo/lofo_summary.csv; "
                "artifacts/llama/lofo/*/model_metadata.json"
            ),
            "takeaway": (
                "Geometry prediction generalizes well within represented families but degrades sharply under LOFO; "
                "zero-shot compilation is heterogeneous and harmful on macro average relative to the global fixed default."
            ),
            "caption": (
                "Generalization beyond represented learning families. (A) Geometry prediction error for held-out "
                "episodes from learning families represented during meta-training versus leave-one-family-out "
                "(LOFO) evaluation, where the test objective is excluded entirely from both training and validation. "
                "(B) Change in oracle regret of the LOFO compiler relative to the global-fixed policy; positive "
                "values indicate that zero-shot compilation improves selection, while negative values indicate worse "
                "selection. Although geometry prediction generalizes well to unseen episodes within represented "
                "families, zero-shot transfer to unseen learning families is not reliably beneficial."
            ),
        },
        {
            "filename": "appendix/fig_exp5_ranking_generalization.pdf, appendix/fig_exp5_ranking_generalization.png",
            "placement": "Appendix",
            "question": "How does within-episode program-ranking fidelity change under LOFO?",
            "panels": "A: represented-family versus LOFO Spearman rank correlation. B: represented-family versus LOFO pairwise ranking accuracy.",
            "sources": (
                "artifacts/llama/prediction/ranking_metrics_by_episode.csv; "
                "artifacts/llama/prediction/predictions_test.csv; "
                "artifacts/llama/prediction/model_metadata.json; "
                "artifacts/llama/lofo/lofo_summary.csv; "
                "artifacts/llama/lofo/*/ranking_metrics_by_episode.csv; "
                "artifacts/llama/lofo/*/predictions_test.csv; "
                "artifacts/llama/lofo/*/model_metadata.json"
            ),
            "takeaway": (
                "Within-episode program-ranking fidelity decreases for every objective when the learning family "
                "is held out, with the largest degradation for factual, lexical, and procedural episodes."
            ),
            "caption": (
                "Program-ranking fidelity under leave-one-family-out generalization. Within-episode agreement "
                "between predicted and observed adaptation-program orderings is compared when the learning family "
                "is represented during meta-training versus excluded entirely under leave-one-family-out (LOFO) "
                "evaluation. (A) Spearman rank correlation. (B) Pairwise ranking accuracy. Ranking fidelity "
                "decreases across all five objectives when the learning family is unseen, with particularly "
                "large degradation for factual, lexical, and procedural episodes."
            ),
        },
        {
            "filename": "appendix/fig_ablation_representation.pdf, appendix/fig_ablation_representation.png",
            "placement": "Appendix",
            "question": "Which pre-adaptation representation supports prediction and selection?",
            "panels": "A: mean geometry MAE. B: compiler oracle regret with top-1 labels.",
            "sources": "artifacts/llama/ablations/*; artifacts/llama/prediction",
            "takeaway": "Compact episode representations perform comparably to the full representation and better than single-signal ablations.",
            "caption": (
                "Representation ablations compare the primary predictor under alternative pre-adaptation feature sets. "
                "Bars and points report geometry prediction error and downstream selection regret."
            ),
        },
        {
            "filename": "appendix/fig_selection_by_objective.pdf, appendix/fig_selection_by_objective.png",
            "placement": "Appendix",
            "question": "Where does compiler selection succeed or fail by learning objective?",
            "panels": "A: per-objective regret. B: per-objective top-1 oracle recovery.",
            "sources": "artifacts/llama/geometry/geometry_dataset.csv; artifacts/llama/selection/compiler_evaluation.csv",
            "takeaway": "Per-objective results expose heterogeneity, including objectives where compiler selection remains harder.",
            "caption": (
                "Per-objective selection diagnostics decompose oracle regret and top-1 recovery for fixed baselines "
                "and the compiler across the five learning objectives."
            ),
        },
        {
            "filename": "appendix/fig_utility_specifications.pdf, appendix/fig_utility_specifications.png",
            "placement": "Appendix",
            "question": "How sensitive is selection to utility specification?",
            "panels": "A: regret under each utility. B: compiler switching relative to balanced utility.",
            "sources": "artifacts/llama/selection/utility_specs/*",
            "takeaway": "The same predicted geometry supports selection under multiple utility preferences, with limited program switching.",
            "caption": (
                "Utility-specification analyses re-score the same candidate programs under alternative outcome weights. "
                "The compiler continues to reduce regret relative to fixed baselines without predictor retraining."
            ),
        },
        {
            "filename": "appendix/fig_lofo_generalization.pdf, appendix/fig_lofo_generalization.png",
            "placement": "Appendix",
            "question": "Does the compiler transfer zero-shot to unseen learning families?",
            "panels": "A: held-out objective utility. B: held-out objective regret.",
            "sources": "artifacts/llama/lofo/lofo_summary.csv",
            "takeaway": "LOFO generalization is a transparent limitation: it does not reliably transfer zero-shot to unseen learning families.",
            "caption": (
                "Leave-one-objective-out generalization trains and selects models using four learning objectives, "
                "then evaluates only on the held-out objective. Results show limited zero-shot transfer to unseen "
                "learning families."
            ),
        },
        {
            "filename": "appendix/fig_seed_robustness.pdf, appendix/fig_seed_robustness.png",
            "placement": "Appendix",
            "question": "Are utilities and winners stable across adaptation seeds?",
            "panels": "A: within-configuration utility standard deviation across seeds. B: strict and tie-aware winner stability by objective.",
            "sources": "artifacts/llama/geometry/robustness/geometry_records_3seed.csv; artifacts/llama/geometry/robustness/*",
            "takeaway": "Adaptation stochasticity is visible in both utility values and oracle-winner identity.",
            "caption": (
                "Adaptation stochasticity across programs. (A) Within-configuration utility variation across "
                "three adaptation seeds. (B) Stability of the oracle-optimal program across seeds under strict "
                "and tie-aware criteria."
            ),
        },
        {
            "filename": "appendix/fig_exp1_margins.pdf, appendix/fig_exp1_margins.png",
            "placement": "Appendix",
            "question": "Are oracle winners separated by meaningful margins?",
            "panels": "Single panel: test-episode winner margins with objective-level bootstrap confidence intervals.",
            "sources": "artifacts/llama/geometry/headroom/headroom_top2_margins.csv; artifacts/llama/geometry/geometry_dataset.csv",
            "takeaway": "Oracle winners are usually separated from the runner-up rather than driven by numerical near-ties.",
            "caption": (
                "Oracle winner separation across held-out episodes. Points show the utility difference between "
                "the best and second-best adaptation programs; black markers show objective-level means with "
                "95% bootstrap confidence intervals."
            ),
        },
        {
            "filename": "appendix/fig_exp1_outcome_sensitivity.pdf, appendix/fig_exp1_outcome_sensitivity.png",
            "placement": "Appendix",
            "question": "Do adaptation programs induce distinct behavioral profiles before scalar utility compression?",
            "panels": "Acquisition, transfer, boundedness, and preservation.",
            "sources": "artifacts/llama/geometry/geometry_dataset.csv",
            "takeaway": "Candidate adaptation programs trade off the four behavioral outcomes differently across learning objectives.",
            "caption": (
                "Configuration sensitivity across adaptation outcomes. Mean acquisition, transfer, boundedness, "
                "and preservation are shown for each budget-matched adaptation program across held-out learning "
                "objectives; error bars show 95% bootstrap confidence intervals."
            ),
        },
    ]
    lines = ["# Compiler Figure Manifest", ""]
    for row in rows:
        lines.extend(
            [
                f"## {row['filename']}",
                f"- Intended placement: {row['placement']}",
                f"- Experiment/question: {row['question']}",
                f"- Panels: {row['panels']}",
                f"- Source files: {row['sources']}",
                f"- Visual takeaway: {row['takeaway']}",
                f"- Suggested LaTeX caption draft: {row['caption']}",
                "",
            ]
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot all compiler paper figures.")
    parser.add_argument("--geometry", default="artifacts/llama/geometry/geometry_dataset.csv")
    parser.add_argument("--output_root", default="outputs/release_verification/figures/llama")
    parser.add_argument("--seed", type=int, default=2026, help="Recorded for reproducibility; figure scripts use seed 2026.")
    parser.add_argument("--experiment2_root", default="artifacts/llama")
    parser.add_argument("--experiment3_root", default="artifacts/llama/selection")
    parser.add_argument("--lofo_summary", default="artifacts/llama/lofo/lofo_summary.csv")
    parser.add_argument("--robustness_analysis_root", default="artifacts/llama/geometry/robustness")
    parser.add_argument("--robustness_raw_records", default="artifacts/llama/geometry/robustness/geometry_records_3seed.csv")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    (output_root / "main").mkdir(parents=True, exist_ok=True)
    (output_root / "appendix").mkdir(parents=True, exist_ok=True)
    (output_root / "data").mkdir(parents=True, exist_ok=True)

    script_dir = Path(__file__).resolve().parent
    experiment2_root = Path(args.experiment2_root)
    prediction_root = experiment2_root / "prediction"
    if not prediction_root.exists():
        prediction_root = experiment2_root / "full_auto"
    lofo_root = experiment2_root / "lofo"
    if not lofo_root.exists():
        lofo_root = experiment2_root / "lofo_episode"
    commands = [
        [sys.executable, str(script_dir / "plot_main_exp1_headroom.py"), "--geometry", args.geometry, "--output_root", args.output_root],
        [
            sys.executable,
            str(script_dir / "plot_main_exp2_prediction.py"),
            "--predictions",
            str(prediction_root / "predictions_test.csv"),
            "--metrics_json",
            str(prediction_root / "prediction_metrics.json"),
            "--ranking_json",
            str(prediction_root / "ranking_metrics.json"),
            "--ranking_by_episode",
            str(prediction_root / "ranking_metrics_by_episode.csv"),
            "--model_metadata",
            str(prediction_root / "model_metadata.json"),
            "--output_root",
            args.output_root,
        ],
        [
            sys.executable,
            str(script_dir / "plot_appendix_exp2_by_objective.py"),
            "--predictions",
            str(prediction_root / "predictions_test.csv"),
            "--ranking_json",
            str(prediction_root / "ranking_metrics.json"),
            "--ranking_by_episode",
            str(prediction_root / "ranking_metrics_by_episode.csv"),
            "--model_metadata",
            str(prediction_root / "model_metadata.json"),
            "--output_root",
            args.output_root,
        ],
        [
            sys.executable,
            str(script_dir / "plot_main_exp3_selection.py"),
            "--geometry",
            args.geometry,
            "--experiment3_root",
            args.experiment3_root,
            "--output_root",
            args.output_root,
        ],
        [
            sys.executable,
            str(script_dir / "plot_exp4_ablation.py"),
            "--experiment2_root",
            args.experiment2_root,
            "--output_root",
            args.output_root,
        ],
        [
            sys.executable,
            str(script_dir / "make_exp4_ablation_by_objective_table.py"),
            "--experiment2_root",
            args.experiment2_root,
            "--output_csv",
            str(Path(args.output_root) / "data" / "exp4_ablation_by_objective.csv"),
            "--output_md",
            str(Path(args.output_root) / "data" / "exp4_ablation_by_objective.md"),
        ],
        [
            sys.executable,
            str(script_dir / "plot_exp5_generalization.py"),
            "--represented_predictions",
            str(prediction_root / "predictions_test.csv"),
            "--represented_metadata",
            str(prediction_root / "model_metadata.json"),
            "--lofo_summary",
            str(lofo_root / "lofo_summary.csv"),
            "--lofo_root",
            str(lofo_root),
            "--output_root",
            args.output_root,
        ],
        [
            sys.executable,
            str(script_dir / "plot_appendix_exp5_ranking_generalization.py"),
            "--represented_root",
            str(prediction_root),
            "--lofo_root",
            str(lofo_root),
            "--output_root",
            args.output_root,
        ],
        [
            sys.executable,
            str(script_dir / "plot_appendix_representation_ablation.py"),
            "--experiment2_root",
            args.experiment2_root,
            "--output_root",
            args.output_root,
        ],
        [
            sys.executable,
            str(script_dir / "plot_appendix_per_objective_selection.py"),
            "--geometry",
            args.geometry,
            "--compiler_csv",
            str(Path(args.experiment3_root) / "compiler_evaluation.csv"),
            "--output_root",
            args.output_root,
        ],
        [
            sys.executable,
            str(script_dir / "plot_appendix_utility_specs.py"),
            "--geometry",
            args.geometry,
            "--utility_root",
            str(Path(args.experiment3_root) / "utility_specs"),
            "--output_root",
            args.output_root,
        ],
        [sys.executable, str(script_dir / "plot_appendix_lofo.py"), "--lofo_summary", args.lofo_summary, "--output_root", args.output_root],
        [
            sys.executable,
            str(script_dir / "plot_appendix_exp1_seed_robustness.py"),
            "--raw_records",
            args.robustness_raw_records,
            "--analysis_root",
            args.robustness_analysis_root,
            "--output_root",
            args.output_root,
        ],
        [
            sys.executable,
            str(script_dir / "plot_appendix_exp1_margins.py"),
            "--geometry",
            args.geometry,
            "--output_root",
            args.output_root,
        ],
        [
            sys.executable,
            str(script_dir / "plot_appendix_exp1_outcome_sensitivity.py"),
            "--geometry",
            args.geometry,
            "--output_root",
            args.output_root,
        ],
    ]
    for cmd in commands:
        run_command(cmd)

    manifest_path = output_root / "FIGURE_MANIFEST.md"
    manifest_path.write_text(figure_manifest(output_root), encoding="utf-8")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
