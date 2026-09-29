# Data Artifacts

This directory contains checked-in dataset and manifest artifacts used by the
current results.

## Live Artifacts

- `latent_specs.jsonl`: 600 latent specs, 120 per learning objective.
- `prompt_examples.jsonl`: 20400 generated prompt examples.
- `calibration_manifest.jsonl`: historical 78-row calibration plan.
- `compiler/episode_manifest.jsonl`: generated one-spec adaptation-compiler
  episode manifest.
- `compiler/config_library.jsonl`: generated phase-0 compiler configuration
  library.

Every manifest row should use `train_budget_per_spec` for the number of
training examples selected per latent spec. The `budget` spelling is reserved
for result tables and command-line shorthand, not manifest rows.

The manifest is not meant to be a balanced design matrix. It contains
legacy/extra calibration grids for lexical, factual, and behavioral policy,
plus newer focused `25`-spec calibration grids for causal mapping and
procedural reasoning.

## Historical Reports and Backups

- `validation_report.txt`: text snapshot of the latest dataset validation
  counts.
- `calibration_manifest_report.txt`: text snapshot of the current manifest
  contract.
- `calibration_manifest.jsonl.bak_budget_extend`: backup from an earlier
  manifest budget-extension pass. Keep it for provenance, but do not treat it
  as the active manifest.

Prefer rerunning validators and regression tests over trusting report files by
themselves:

```bash
python src/validate_dataset.py
python -m unittest discover -s tests
```
