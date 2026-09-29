# Release Artifacts

This directory contains immutable, curated evidence for the reported analyses.
Analysis scripts may read these files, but generated files must be written to
`outputs/` or another explicit temporary directory.

The release includes compact geometry records, held-out predictions, selection
evaluations, localization summaries, figure-source tables, and final figures.
Raw per-job outputs, adapters, checkpoints, tokenizer copies, feature payloads,
and trained predictor binaries are intentionally omitted.

Precomputed predictions and all numerical inputs required to reproduce the
reported analyses and figures are included. Recomputing pre-adaptation
representations requires access to the corresponding base model and GPU
resources.

## Directory Map

- `llama/`: current Table 1 calibration plus geometry, prediction, selection,
  ablation, and LOFO evidence.
- `gemma/`: calibration and corrected geometry, prediction, and selection
  evidence.
- `localization/`: primary, cross-model, and parameter-matched evidence.
- `figures/data/`: compact source tables consumed by figure scripts.
- `figures/main/`: released main-text figures.
- `figures/appendix/`: released appendix figures.
- `MANIFEST.csv`: sizes, SHA-256 hashes, roles, provenance, scorer versions,
  and regeneration commands for every curated evidence file.

## Provenance Conventions

`<ARCHIVE_ROOT>` denotes the private, unreleased experiment-working archive
from which a compact artifact was derived. The suffix after that placeholder is
preserved for historical lineage; it is not expected to resolve in this tree.

Llama compiler artifacts use the current authoritative scorer state. Gemma
compiler artifacts originate from the corrected `causal_content_v1` lineage.
The Gemma causal rescore manifest records the relationship to omitted raw and
corrected evaluation files without requiring the full evaluation mirror.

The current-paper calibration evidence is split deliberately: the four Llama
episode tables in `llama/calibration/` reconstruct Table 1, while the
independently calibrated Gemma files in `gemma/calibration/` support Table 12.
Earlier localization calibration plots and transferred-budget sensitivity
outputs are historical and are not included.

## Figure and Table Reproduction

Run the CPU commands documented in `docs/reproducibility.md`. They read only
included files under `artifacts/` and write regenerated material beneath
`outputs/release_verification/`; they do not modify this directory.
