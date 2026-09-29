# Scripts Guide

`scripts/` contains runnable research workflows. Shared implementation remains
in `src/`; these scripts are entry points and orchestration helpers rather than
a distributable Python package.

## Reproduce

`reproduce/` regenerates paper figures, tables, statistical summaries, and
release-verification outputs from checked-in artifacts. The compiler-specific
plotting modules live together in `reproduce/compiler/` because they share
local plotting helpers.

The lightweight release commands are documented in
`docs/reproducibility.md`.

## Experiments

- `experiments/calibration/`: calibration, schedule selection, and budget
  sweeps.
- `experiments/localization/`: localization launchers, controls, and result
  summarization.
- `experiments/compiler/`: compiler geometry runs, feature extraction,
  predictor training, and the Gemma replication workflow.

Experiment launchers can require model downloads and accelerator resources.
Use their dry-run modes where available before starting a full run.

## Audits

`audits/` contains dataset, scorer, output, and conservative-rejection checks,
plus narrowly scoped rescoring utilities. These tools validate existing inputs
or outputs; they are not paper-figure generators.

## Maintenance

`maintenance/` contains curated-manifest and release-tree utilities. In
particular, `make_artifact_manifest.py` rebuilds `artifacts/MANIFEST.csv` after
intentional artifact metadata or path-reference changes.

Use maintenance scripts carefully: the checked-in manifests are release
artifacts, not disposable build outputs.
