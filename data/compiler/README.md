# Compiler Data

This directory contains generated adaptation-compiler inputs. These files are
derived from the canonical AAAI dataset but are kept separate from it.

- `episode_manifest.jsonl`: one latent specification per adaptation episode,
  with deterministic `train`/`validation`/`test` meta splits at the `spec_id`
  level.
- `config_library.jsonl`: phase-0 candidate LoRA configurations:
  `full`, `early`, `middle`, and `late`.
- `preservation_examples.jsonl`: fixed unrelated prompt-target pool used to
  measure preservation of frozen-correct behavior after adaptation. This raw
  pool is not model-filtered; cache frozen-correct subsets with
  `src/evaluate_preservation.py`.

Regenerate with:

```bash
python src/make_compiler_episode_manifest.py
python src/make_compiler_config_library.py
python src/make_preservation_dataset.py
```

Do not replace `data/prompt_examples.jsonl` or
`data/calibration_manifest.jsonl` with these files.
