# Methodology Note

This benchmark studies localized learning: whether adaptation can acquire a new
piece of knowledge or behavior while keeping the change constrained to selected
regions of the model.

The repository separates three concepts:

- **Latent specs**: abstract things the model should learn.
- **Prompt examples**: train/eval/control prompts generated from each spec.
- **Calibration manifest rows**: concrete experiment plans selecting specs and
  training budgets.

## Learning Objectives

### Lexical Binding

A lexical spec maps a synthetic word to a familiar concept:

```json
{
  "learning_type": "lexical_binding",
  "novel_term": "braelzen",
  "target_concept": "bicycle",
  "category": "vehicle",
  "attributes": ["two wheels", "pedals", "handlebars"],
  "affordances": ["can be ridden to work", "can be locked outside"]
}
```

The generalization split asks about attributes, categories, and affordances so
success requires using the learned binding, not only repeating the target word.

### Factual Association

A factual spec defines a synthetic fact or relation:

```json
{
  "learning_type": "factual_association",
  "subject": "Norland",
  "relation": "capital_city",
  "object": "Vespra"
}
```

The generalization split asks how the fact should be used in a downstream
context, not only direct recall.

### Behavioral Policy

A behavioral-policy spec defines which action, response style, or preference
should be selected in a synthetic situation. These examples test whether an
adaptation can learn a policy-like regularity rather than a named entity.

### Causal Mapping

A causal-mapping spec defines a synthetic cause/effect relationship. These
examples test whether adaptation can learn directional structure and apply it to
counterfactual or diagnostic prompts.

### Procedural Reasoning

A procedural-reasoning spec defines a small synthetic procedure. These examples
test whether adaptation can learn and apply a multi-step transformation rather
than retrieve a single fact.

## Splits

Each latent spec currently generates:

- `train`: supervised adaptation examples.
- `id_eval`: direct in-distribution checks.
- `paraphrase_eval`: surface paraphrases of the learned item.
- `generalization`: prompts that require applying the learned item.
- `negative_control`: distractor prompts where the model should reject an
  incorrect association.

The current checked-in dataset has `120` specs per learning objective and this
per-spec split structure:

| Split | Examples per spec |
| --- | ---: |
| `train` | 12 |
| `id_eval` | 5 |
| `paraphrase_eval` | 5 |
| `generalization` | 6 |
| `negative_control` | 6 |

## Scoring

Scoring is intentionally stricter than plain string matching.

- Positive yes/no prompts need correct polarity for loose score and explicit
  target mention for strict score.
- Negative controls pass when the response starts with a rejection, even if it
  names the distractor in the rejection.
- Direct recall prompts require the exact target and reject negated mentions.
- Objective-specific evaluators add fields needed for behavioral, causal, and
  procedural output audits.

Shared lexical/factual scoring helpers live in `src/common.py`. Specialized
evaluators live in the `src/evaluate_calibration_run_*.py` files.

## Calibration Before Localization

Equal example count does not imply equal difficulty. Calibration finds budgets
where objectives are learnable enough to compare, but not so saturated that
localized effects are hidden.

The current final cross-objective localization comparison uses:

| Objective | Calibrated budget |
| --- | ---: |
| Lexical binding | 10 |
| Factual association | 8 |
| Behavioral policy | 10 |
| Causal mapping | 10 |
| Procedural reasoning | 8 |

The checked-in manifest is a historical calibration manifest rather than a
balanced design matrix. It contains legacy/extra calibration grids for lexical,
factual, and behavioral policy, plus the newer focused `25`-spec calibration
grids for causal mapping and procedural reasoning.

Only after this calibration should localized adaptation profiles be interpreted
as evidence of different learning signatures.