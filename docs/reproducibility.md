# Reproducibility Guide

The release supports two distinct workflows: CPU reproduction from included
compact artifacts, and full model-backed reproduction of adaptation and
feature-generation experiments.

## CPU Environment and Validation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m unittest discover -s tests
python src/validate_dataset.py
```

No model download is required for these checks.

## Figure and Table Command Matrix

All commands below read immutable inputs from `artifacts/` and write beneath
`outputs/release_verification/` unless an explicit output override is supplied.

| Evidence family | Command | Expected output family |
|---|---|---|
| Llama schedule calibration (Table 1) | `python scripts/reproduce/compiler/make_llama_calibration_table.py` | `outputs/release_verification/tables/llama/table1_calibration.csv` |
| Llama compiler main/appendix figures and tables | `python scripts/reproduce/compiler/plot_all_paper_figures.py` | `outputs/release_verification/figures/llama/` |
| Gemma compiler figures | `python scripts/reproduce/plot_exp6_gemma.py` | `outputs/release_verification/figures/gemma/` |
| Cross-experiment scope synthesis | `python scripts/reproduce/plot_compilation_scope.py` | `outputs/release_verification/figures/synthesis/` |
| Cross-model localization figures | `python scripts/reproduce/make_cross_model_figures.py` | `outputs/release_verification/figures/localization/cross_model/` |
| Primary localization table | `python scripts/reproduce/make_cross_objective_localization_table.py` | `outputs/release_verification/tables/localization/` |
| Primary localization figure | `python scripts/reproduce/make_cross_objective_localization_figure.py` | `outputs/release_verification/figures/localization/` |

The release verification compares numerical source tables and successful
generation. PDF bytes need not be identical because generator metadata may
vary.

The Llama Table 1 command averages the four 25-episode files in
`artifacts/llama/calibration/`. Current Table 12 is independently verified from
`artifacts/gemma/calibration/calibration_summary.csv`. Historical localization
calibration plots and the transferred-budget Gemma sensitivity analysis are
not current-paper outputs and are intentionally excluded.

## Included Evidence Boundary

Precomputed predictions and all numerical inputs required to reproduce the
reported analyses and figures are included. Recomputing pre-adaptation
representations requires access to the corresponding base model and GPU
resources.

Raw job trees, trained adapters, checkpoints, tokenizer copies, predictor
binaries, and feature payloads are not part of the canonical release.
`<ARCHIVE_ROOT>` in metadata denotes those omitted historical working files; it
is not a path required by the CPU workflows.

## Full GPU/Model Reproduction

Install `requirements-gpu.txt`, obtain model access through the model provider,
and use the calibration/training commands documented in
`docs/how_to_run_calibration.md` and `docs/adaptation_compiler.md`. Direct every
new run to `outputs/` and retain CLI path overrides when using an external
archive.

GPU workflows include adaptation, evaluation, feature extraction, and optional
predictor retraining. Their runtime and memory requirements depend on model,
precision, and adaptation configuration. They are deliberately excluded from
the normal CPU test suite.