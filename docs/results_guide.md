# Released Evidence Guide

All included numerical evidence and paper-facing figures live under
`artifacts/`. Newly generated local work belongs under the ignored `outputs/`
directory.

## Llama Compiler Evidence

- `artifacts/llama/calibration/`: four 25-episode optimization-schedule tables
  that reconstruct current Table 1.
- `artifacts/llama/geometry/`: seed-level and aggregated geometry, headroom,
  and seed-robustness analyses.
- `artifacts/llama/prediction/`: held-out predictions and prediction/ranking
  metrics.
- `artifacts/llama/selection/`: compiler evaluations and utility-specification
  sensitivity.
- `artifacts/llama/ablations/`: episode, frozen-behavior, and module-probe
  representation ablations.
- `artifacts/llama/lofo/`: five leave-one-family-out evaluations.

## Gemma Compiler Evidence

- `artifacts/gemma/calibration/`: the independent Gemma schedule calibration
  supporting current Table 12.
- `artifacts/gemma/geometry/`: corrected seed-level and aggregated geometry,
  headroom, and causal-rescore lineage.
- `artifacts/gemma/prediction/`: held-out predictions and ranking metrics.
- `artifacts/gemma/selection/`: compiler-selection evaluations.

The authoritative Gemma lineage is `causal_content_v1`. The full corrected
evaluation mirror is intentionally omitted; its compact provenance is retained
in `artifacts/gemma/geometry/causal_rescore_manifest.csv`.

## Localization Evidence

- `artifacts/localization/primary/`: primary cross-objective localization
  tables, statistics, mislocation analyses, and audits.
- `artifacts/localization/cross_model/`: cross-model summaries and geometry.
- `artifacts/localization/parameter_matched/`: parameter-matched controls.

Historical localization calibration plots and the earlier transferred-budget
Gemma sensitivity study are not part of the current-paper artifact inventory.

## Figures

`artifacts/figures/data/` contains the compact source tables used for released
figures. Final main and appendix figures are under `artifacts/figures/main/`
and `artifacts/figures/appendix/`. Cross-experiment compilation-scope material
is namespaced under `synthesis/`.

## Provenance

`artifacts/MANIFEST.csv` supplies size and SHA-256 verification for every
curated evidence file. `<ARCHIVE_ROOT>` fields identify intentionally omitted
historical raw sources; figure/table reproduction does not require them.
