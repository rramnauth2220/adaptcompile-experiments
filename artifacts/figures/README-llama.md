# Compiler Figure Manifest

## main/fig_exp1_headroom.pdf, main/fig_exp1_headroom.png
- Intended placement: Main
- Experiment/question: Do different episodes prefer different adaptation programs?
- Panels: A: tie-aware oracle program shares by objective. B: objective-fixed regret/headroom by episode on the same objective axis.
- Source files: artifacts/llama/geometry/geometry_dataset.csv
- Visual takeaway: Oracle-optimal programs vary across learning objectives, and many held-out episodes retain measurable episode-specific headroom beyond objective-fixed selection.
- Suggested LaTeX caption draft: Experiment 1 shows selection headroom in the candidate adaptation-program library. Panel A reports tie-aware oracle-optimal program shares by learning objective; Panel B shows episode-level utility gained by oracle selection relative to a train-derived objective-fixed baseline on the same objective axis.

## main/fig_exp2_prediction.pdf, main/fig_exp2_prediction.png
- Intended placement: Main
- Experiment/question: Can pre-adaptation signals predict adaptation geometry?
- Panels: A: predicted versus observed utility. B: geometry prediction error. C: within-episode program ranking.
- Source files: artifacts/llama/prediction/*
- Visual takeaway: The primary predictor tracks observed utility, improves outcome-level geometry prediction, and better preserves within-episode program order.
- Suggested LaTeX caption draft: Predicting adaptation geometry from pre-adaptation episode--model signals. (A) Predicted and observed balanced utility across held-out episode--program pairs; the dashed line indicates perfect agreement. (B) Mean absolute error for acquisition, transfer, boundedness, and preservation under the primary geometry predictor and non-episode-conditioned baselines. (C) Within-episode program ranking measured by Spearman correlation, pairwise ranking accuracy, and tie-aware top-1 and top-2 oracle recovery.

## appendix/fig_exp2_by_objective.pdf, appendix/fig_exp2_by_objective.png
- Intended placement: Appendix
- Experiment/question: Is Experiment 2 prediction performance distributed across learning objectives?
- Panels: A: episode-level geometry prediction error by objective. B: within-episode program-ranking fidelity by objective.
- Source files: artifacts/llama/prediction/predictions_test.csv; artifacts/llama/prediction/ranking_metrics_by_episode.csv; artifacts/llama/prediction/ranking_metrics.json; artifacts/llama/prediction/model_metadata.json
- Visual takeaway: The primary geometry predictor remains informative across represented learning objectives while exposing objective-level variation in prediction error and ranking fidelity.
- Suggested LaTeX caption draft: Geometry prediction across learning objectives. (A) Episode-level mean absolute error between predicted and observed adaptation geometry, grouped by learning objective. Points denote held-out episodes and summary markers show means with 95% bootstrap confidence intervals. (B) Within-episode agreement between predicted and observed program orderings, measured by Spearman rank correlation and pairwise ranking accuracy. The same primary predictor is used for all objectives and is not given objective identity.

## main/fig_exp3_selection.pdf, main/fig_exp3_selection.png
- Intended placement: Main
- Experiment/question: Does predicted geometry improve program selection?
- Panels: A: realized utility for fixed baselines, compiler, and oracle. B: regret under utility specifications.
- Source files: artifacts/llama/geometry/geometry_dataset.csv; artifacts/llama/selection/*
- Visual takeaway: The compiler reduces regret relative to fixed baselines and approaches the oracle under several utility specifications.
- Suggested LaTeX caption draft: Experiment 3 turns predicted geometry into a single executable adaptation-program choice. The compiler reduces oracle regret relative to global and objective-fixed baselines, while the oracle marks the exhaustive-search ceiling.

## main/fig_exp4_ablation.pdf, main/fig_exp4_ablation.png
- Intended placement: Main
- Experiment/question: Which pre-adaptation representation is sufficient for compilation?
- Panels: A: geometry MAE. B: compiler oracle regret. C: tie-aware top-1 oracle recovery.
- Source files: <ARCHIVE_ROOT>/results/compiler/experiment2/ablation_*; artifacts/llama/prediction
- Visual takeaway: The episode-only representation retains performance comparable to the full representation, while module-probe and frozen-behavior features are weaker when used independently.
- Suggested LaTeX caption draft: Representation ablations. Geometry prediction and downstream selection are compared across the full episode--model representation and three restricted feature sets. (A) Mean adaptation-geometry prediction error. (B) Oracle regret of the program selected from predicted geometry. (C) Tie-aware top-1 recovery of an oracle-optimal program. The episode representation alone retains performance comparable to the full representation, while module-level probes and frozen-model behavioral statistics are weaker when used independently.

## main/fig_exp5_generalization.pdf, main/fig_exp5_generalization.png
- Intended placement: Main
- Experiment/question: Does the adaptation compiler generalize beyond represented learning families?
- Panels: A: represented-family versus LOFO geometry prediction error. B: LOFO regret reduction relative to the global-fixed policy.
- Source files: artifacts/llama/prediction/predictions_test.csv; artifacts/llama/prediction/model_metadata.json; artifacts/llama/lofo/lofo_summary.csv; artifacts/llama/lofo/*/model_metadata.json
- Visual takeaway: Geometry prediction generalizes well within represented families but degrades sharply under LOFO; zero-shot compilation is heterogeneous and harmful on macro average relative to the global fixed default.
- Suggested LaTeX caption draft: Generalization beyond represented learning families. (A) Geometry prediction error for held-out episodes from learning families represented during meta-training versus leave-one-family-out (LOFO) evaluation, where the test objective is excluded entirely from both training and validation. (B) Change in oracle regret of the LOFO compiler relative to the global-fixed policy; positive values indicate that zero-shot compilation improves selection, while negative values indicate worse selection. Although geometry prediction generalizes well to unseen episodes within represented families, zero-shot transfer to unseen learning families is not reliably beneficial.

## appendix/fig_exp5_ranking_generalization.pdf, appendix/fig_exp5_ranking_generalization.png
- Intended placement: Appendix
- Experiment/question: How does within-episode program-ranking fidelity change under LOFO?
- Panels: A: represented-family versus LOFO Spearman rank correlation. B: represented-family versus LOFO pairwise ranking accuracy.
- Source files: artifacts/llama/prediction/ranking_metrics_by_episode.csv; artifacts/llama/prediction/predictions_test.csv; artifacts/llama/prediction/model_metadata.json; artifacts/llama/lofo/lofo_summary.csv; artifacts/llama/lofo/*/ranking_metrics_by_episode.csv; artifacts/llama/lofo/*/predictions_test.csv; artifacts/llama/lofo/*/model_metadata.json
- Visual takeaway: Within-episode program-ranking fidelity decreases for every objective when the learning family is held out, with the largest degradation for factual, lexical, and procedural episodes.
- Suggested LaTeX caption draft: Program-ranking fidelity under leave-one-family-out generalization. Within-episode agreement between predicted and observed adaptation-program orderings is compared when the learning family is represented during meta-training versus excluded entirely under leave-one-family-out (LOFO) evaluation. (A) Spearman rank correlation. (B) Pairwise ranking accuracy. Ranking fidelity decreases across all five objectives when the learning family is unseen, with particularly large degradation for factual, lexical, and procedural episodes.

## appendix/fig_ablation_representation.pdf, appendix/fig_ablation_representation.png
- Intended placement: Appendix
- Experiment/question: Which pre-adaptation representation supports prediction and selection?
- Panels: A: mean geometry MAE. B: compiler oracle regret with top-1 labels.
- Source files: <ARCHIVE_ROOT>/results/compiler/experiment2/ablation_*; artifacts/llama/prediction
- Visual takeaway: Compact episode representations perform comparably to the full representation and better than single-signal ablations.
- Suggested LaTeX caption draft: Representation ablations compare the primary predictor under alternative pre-adaptation feature sets. Bars and points report geometry prediction error and downstream selection regret.

## appendix/fig_selection_by_objective.pdf, appendix/fig_selection_by_objective.png
- Intended placement: Appendix
- Experiment/question: Where does compiler selection succeed or fail by learning objective?
- Panels: A: per-objective regret. B: per-objective top-1 oracle recovery.
- Source files: artifacts/llama/geometry/geometry_dataset.csv; artifacts/llama/selection/compiler_evaluation.csv
- Visual takeaway: Per-objective results expose heterogeneity, including objectives where compiler selection remains harder.
- Suggested LaTeX caption draft: Per-objective selection diagnostics decompose oracle regret and top-1 recovery for fixed baselines and the compiler across the five learning objectives.

## appendix/fig_utility_specifications.pdf, appendix/fig_utility_specifications.png
- Intended placement: Appendix
- Experiment/question: How sensitive is selection to utility specification?
- Panels: A: regret under each utility. B: compiler switching relative to balanced utility.
- Source files: artifacts/llama/selection/utility_specs/*
- Visual takeaway: The same predicted geometry supports selection under multiple utility preferences, with limited program switching.
- Suggested LaTeX caption draft: Utility-specification analyses re-score the same candidate programs under alternative outcome weights. The compiler continues to reduce regret relative to fixed baselines without predictor retraining.

## appendix/fig_lofo_generalization.pdf, appendix/fig_lofo_generalization.png
- Intended placement: Appendix
- Experiment/question: Does the compiler transfer zero-shot to unseen learning families?
- Panels: A: held-out objective utility. B: held-out objective regret.
- Source files: artifacts/llama/lofo/lofo_summary.csv
- Visual takeaway: LOFO generalization is a transparent limitation: it does not reliably transfer zero-shot to unseen learning families.
- Suggested LaTeX caption draft: Leave-one-objective-out generalization trains and selects models using four learning objectives, then evaluates only on the held-out objective. Results show limited zero-shot transfer to unseen learning families.

## appendix/fig_seed_robustness.pdf, appendix/fig_seed_robustness.png
- Intended placement: Appendix
- Experiment/question: Are utilities and winners stable across adaptation seeds?
- Panels: A: within-configuration utility standard deviation across seeds. B: strict and tie-aware winner stability by objective.
- Source files: artifacts/llama/geometry/robustness/geometry_records_3seed.csv; artifacts/llama/geometry/robustness/*
- Visual takeaway: Adaptation stochasticity is visible in both utility values and oracle-winner identity.
- Suggested LaTeX caption draft: Adaptation stochasticity across programs. (A) Within-configuration utility variation across three adaptation seeds. (B) Stability of the oracle-optimal program across seeds under strict and tie-aware criteria.

## appendix/fig_exp1_margins.pdf, appendix/fig_exp1_margins.png
- Intended placement: Appendix
- Experiment/question: Are oracle winners separated by meaningful margins?
- Panels: Single panel: test-episode winner margins with objective-level bootstrap confidence intervals.
- Source files: artifacts/llama/geometry/headroom/headroom_top2_margins.csv; artifacts/llama/geometry/geometry_dataset.csv
- Visual takeaway: Oracle winners are usually separated from the runner-up rather than driven by numerical near-ties.
- Suggested LaTeX caption draft: Oracle winner separation across held-out episodes. Points show the utility difference between the best and second-best adaptation programs; black markers show objective-level means with 95% bootstrap confidence intervals.

## appendix/fig_exp1_outcome_sensitivity.pdf, appendix/fig_exp1_outcome_sensitivity.png
- Intended placement: Appendix
- Experiment/question: Do adaptation programs induce distinct behavioral profiles before scalar utility compression?
- Panels: Acquisition, transfer, boundedness, and preservation.
- Source files: artifacts/llama/geometry/geometry_dataset.csv
- Visual takeaway: Candidate adaptation programs trade off the four behavioral outcomes differently across learning objectives.
- Suggested LaTeX caption draft: Configuration sensitivity across adaptation outcomes. Mean acquisition, transfer, boundedness, and preservation are shown for each budget-matched adaptation program across held-out learning objectives; error bars show 95% bootstrap confidence intervals.
