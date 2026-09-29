# Gemma Compiler Replication Data

This directory holds Gemma-specific compiler protocol files. It does not replace
or modify the frozen Llama compiler inputs in `data/compiler/`.

- `config_library.jsonl` defines the Gemma adaptation programs.
- Gemma-2-9B-IT is treated as a 42-layer backbone.
- Localized programs use normalized 25% depth windows, so early/middle/late
  adapt 11 contiguous layers.
- The primary budget-matched programs are `early__all__r16`,
  `middle__all__r16`, `late__all__r16`, and `full__all__r4`.
- `full__all__r16` is retained as the high-capacity calibration/baseline
  program.

The episode manifest and prompt examples are intentionally reused from
`data/compiler/episode_manifest.jsonl` and `data/prompt_examples.jsonl` so Llama
and Gemma can be compared on matched latent specifications.

Gemma calibration writes `outputs/compiler_gemma/calibration/selected_schedule.json`;
the selected release copy is under `artifacts/gemma/calibration/`.
If that file has `"status": "inconclusive"`, pilot/full orchestration stops
before launching adaptation jobs. After inspecting `calibration_summary.csv`,
either rerun the calibration summary with explicit gates or pass
`--force_gradient_accumulation_steps <N>` to `scripts/experiments/compiler/run_compiler_gemma_replication.py`
to record an intentional manual schedule choice.
