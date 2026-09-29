# Adaptation Compilation

> **TL;DR:** This repository accompanies *Compiling Learning Problems into Adaptation Programs for Language Models*. It contains the datasets, code, and released evidence for predicting how a learning episode will respond to candidate LoRA programs - across acquisition, transfer, boundedness, and preservation - and selecting a program before adaptation begins.

Adaptation compilation treats the choice of **where and how a model should update** as a prediction and decision problem. Rather than using one fixed adaptation recipe or searching from scratch for every episode, the compiler learns from prior adaptations, predicts the behavioral tradeoffs of each candidate program, and selects one for the desired utility.

The primary experiments use Llama-3.1-8B-Instruct; a Gemma-2-9B-IT replication tests how the selection problem and compiler behavior change across backbones.

## Navigate the repository

| Path | What it contains |
|---|---|
| [`docs/`](docs/) | Methodology, result interpretation, and reproduction guides |
| [`data/`](data/) | Canonical synthetic datasets, manifests, and compiler configuration libraries |
| [`artifacts/`](artifacts/) | Immutable released evidence, source tables, and paper figures |
| [`src/`](src/) | Dataset, evaluation, geometry, feature-extraction, prediction, and selection code |
| [`scripts/reproduce/`](scripts/reproduce/) | CPU-friendly figure, table, and analysis entry points |
| [`scripts/experiments/`](scripts/experiments/) | Model-backed calibration, localization, and compiler workflows |
| [`scripts/audits/`](scripts/audits/) | Dataset, scorer, and output checks |
| [`tests/`](tests/) | CPU tests for data, scorers, artifacts, and release integrity |

Checked-in evidence under `artifacts/` should not be overwritten. New runs and regenerated outputs belong under `outputs/`.

## Quick start

The lightweight path validates the release and does not download models:

```bash
python -m venv .venv
# Activate the environment, then:
python -m pip install -r requirements.txt
python -m unittest discover -s tests
python src/validate_dataset.py
```

To regenerate paper figures and tables from the included compact artifacts, follow [`docs/reproducibility.md`](docs/reproducibility.md). For full model-backed runs, install `requirements-gpu.txt` and start with [`docs/adaptation_compiler.md`](docs/adaptation_compiler.md) or [`docs/how_to_run_calibration.md`](docs/how_to_run_calibration.md).

## Suggested reading order

1. [`docs/repository_guide.md`](docs/repository_guide.md) - release layout and data boundaries.
2. [`docs/results_guide.md`](docs/results_guide.md) - where each reported result lives.
3. [`docs/methodology_note.md`](docs/methodology_note.md) - metrics and interpretation.
4. [`scripts/README.md`](scripts/README.md) and [`src/README.md`](src/README.md) - runnable workflows and implementation map.

## License

MIT. See [`LICENSE`](LICENSE).
