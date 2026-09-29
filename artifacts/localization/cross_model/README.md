# Cross-Model Localization Results

This directory contains the cross-model robustness analysis for localized LoRA
adaptation. The experiment asks whether the adaptation-geometry profiles found
in the Llama-3.1-8B-Instruct experiments replicate across model families when
the same selected objective budgets are reused.

## Data Included

The current summary contains:

- 5 model families:
  - `meta-llama/Llama-3.1-8B-Instruct`
  - `mistralai/Mistral-7B-Instruct-v0.3`
  - `google/gemma-2-9b-it`
  - `allenai/OLMo-2-1124-7B-Instruct`
  - `Qwen/Qwen2.5-14B-Instruct`
- 5 learning objectives:
  - lexical binding
  - factual association
  - behavioral policy
  - causal mapping
  - procedural reasoning
- 4 localization conditions:
  - full
  - early
  - middle
  - late
- 3 seeds per model-objective-condition.

The selected training budgets are the budgets chosen from the original Llama
calibration experiments:

| Objective | Budget |
|---|---:|
| lexical binding | 10 |
| factual association | 8 |
| behavioral policy | 10 |
| causal mapping | 10 |
| procedural reasoning | 8 |

These are not model-specific budgets. This is important for interpretation:
cross-model differences may reflect genuine architectural differences, but they
may also reflect that a Llama-selected budget is too small or otherwise
mismatched for another model family.

## Core Metrics

For each run, the cross-model summary uses:

- acquisition: mean of strict accuracy on `id_eval` and `paraphrase_eval`
- transfer: strict accuracy on `generalization`
- boundedness: objective-specific negative-control score
  - lexical binding: strict accuracy on `negative_control`
  - factual association: strict accuracy on `negative_control`
  - behavioral policy: concept accuracy on `negative_control`
  - causal mapping: concept accuracy on `negative_control`
  - procedural reasoning: concept accuracy on `negative_control`

Higher is better for all three metrics. Boundedness should be interpreted as
negative-control preservation rather than task acquisition.

## Files

Top-level summaries:

- `summary_cross_model_by_seed.csv`: one row per
  model/objective/condition/seed.
- `summary_cross_model_aggregate.csv`: means and standard deviations across
  seeds for each model/objective/condition.

Geometry analysis:

- `geometry_analysis/cross_model_geometry_profiles_by_seed.csv`
- `geometry_analysis/cross_model_best_regions.csv`
- `geometry_analysis/cross_model_best_region_agreement.csv`
- `geometry_analysis/cross_model_geometry_similarity.csv`
- `geometry_analysis/cross_model_geometry_similarity_summary.csv`
- `geometry_analysis/cross_model_variance_decomposition.json`
- `geometry_analysis/cross_model_permutation_tests.json`
- `geometry_analysis/cross_model_mislocation_penalties_by_seed.csv`
- `geometry_analysis/cross_model_mislocation_sign_agreement.csv`
- `geometry_analysis/cross_model_geometry_report.md`

Figures:

- `<ARCHIVE_ROOT>/figures/cross_model_full_stack_performance.{svg,png,pdf}`
- `<ARCHIVE_ROOT>/figures/cross_model_delta_geometry_heatmap.{svg,png,pdf}`
- `<ARCHIVE_ROOT>/figures/cross_model_best_region_agreement.{svg,png,pdf}`
- `<ARCHIVE_ROOT>/figures/cross_model_primary_mislocation.{svg,png,pdf}`
- `<ARCHIVE_ROOT>/figures/cross_model_similarity_variance.{svg,png,pdf}`

Figure companion tables:

- `<ARCHIVE_ROOT>/figures/cross_model_full_stack_performance.csv`
- `<ARCHIVE_ROOT>/figures/cross_model_calibration_mismatch_diagnostics.csv`
- `<ARCHIVE_ROOT>/figures/cross_model_delta_geometry_stats.csv`
- `<ARCHIVE_ROOT>/figures/cross_model_figure_manifest.csv`
- `<ARCHIVE_ROOT>/figures/cross_model_figure_includes.tex`

The SVG files are the source figures. The PNG files are high-resolution raster
exports for slides and quick inspection. The PDF files are the preferred
LaTeX/paper includes.

## Statistical Notation and Tests

### Localized-Minus-Full Geometry

For model `M`, objective `T`, seed `s`, and metric `m`, define:

```text
G_{M,T,s,c,m}
```

as the metric value for condition `c`, where `c` is one of `full`, `early`,
`middle`, or `late`.

The localized-minus-full effect is:

```text
Delta_{M,T,s,c,m} = G_{M,T,s,c,m} - G_{M,T,s,full,m}
```

for `c` in `{early, middle, late}`.

The 9-dimensional geometry profile for each model, objective, and seed is:

```text
DeltaG_{M,T,s} =
[
  Delta_acquisition_early,
  Delta_acquisition_middle,
  Delta_acquisition_late,
  Delta_transfer_early,
  Delta_transfer_middle,
  Delta_transfer_late,
  Delta_boundedness_early,
  Delta_boundedness_middle,
  Delta_boundedness_late
]
```

In `cross_model_delta_geometry_heatmap.svg`, each cell reports the mean of
`Delta` across model-seed pairs. With 5 models and 3 seeds, the nominal sample
size is `n = 15` paired deltas per objective/metric/condition cell. The `+/-`
value is a two-sided 95% confidence interval computed as:

```text
mean(Delta) +/- t_{0.975, n-1} * sd(Delta) / sqrt(n)
```

An asterisk marks cells where the 95% CI excludes 0. These annotations are
descriptive and are not corrected for multiple comparisons.

Display precision is chosen for readability: heatmap deltas and full-stack
performance cells are shown as percentages or percentage points to one decimal
place, geometry cosine summaries are shown to two decimals, and variance ratios
are shown to three decimals in text or whole percentages in stacked bars.

### Best Localized Region Agreement

For each model, objective, and metric, the best localized condition is the
highest mean value among `early`, `middle`, and `late`, averaging across seeds.
The agreement table then asks which localized region is most often selected
across the five models.

Notation:

```text
agreement_rate = count(most_common_best_region) / n_models
```

In `cross_model_best_region_agreement.svg`, each cell reports the modal
localized region and agreement as `k/N`.

### Geometry Similarity

For each objective, the analysis computes each model's mean 9-dimensional
`DeltaG` profile across seeds. It then compares models with:

- cosine similarity
- Euclidean distance

For each objective, there are 10 model pairs from 5 models. The figure reports
mean cosine similarity with `+/- SD` across these model pairs.

### Variance Decomposition

The variance decomposition is descriptive. It uses the seed-level `DeltaG`
profiles and computes additive sums of squares for:

- objective means
- model means
- residual / objective-by-model interaction proxy

The current result is:

```text
objective explained ratio = 0.250
model explained ratio     = 0.342
```

Thus, in this cross-model run, the model component is larger than the objective
component. This does not invalidate the localization results, but it does mean
the current cross-model result should not be framed as "objective explains more
geometry variation than model." A more conservative interpretation is that
objective-specific geometry signatures replicate for several planned contrasts,
but model family still contributes substantial variation.

### Permutation Tests

Two label-permutation tests are reported:

- objective-label permutation: tests whether objective labels explain geometry
  profiles more than expected by chance.
- model-label permutation: tests whether model labels explain geometry profiles
  more than expected by chance.

The test statistic is between-group sum of squares in the 9-dimensional
`DeltaG` space. Labels are randomly permuted and the p-value is:

```text
p = (number of permuted statistics >= observed statistic + 1) / (n_perm + 1)
```

With `n_perm = 10000`, the minimum possible p-value is approximately `0.0001`.
Both objective and model labels are significant in the current analysis,
consistent with both task type and model family carrying geometry information.

### Primary Mislocation Sign Tests

The primary planned contrasts are:

| Objective | Primary metric | Preferred > mislocated |
|---|---|---|
| lexical binding | acquisition | early > late |
| factual association | transfer | late > early |
| behavioral policy | boundedness | middle > late |
| causal mapping | transfer | middle > early |
| procedural reasoning | transfer | middle > early |

For each objective, model-level deltas are computed by averaging across seeds:

```text
delta_model = mean_s(G_preferred - G_mislocated)
```

The figure reports:

- model-level points
- mean model-level delta
- 95% CI across model-level deltas
- one-sided sign-test p-value under `H0: P(delta > 0) = 0.5`

In `cross_model_primary_mislocation`, the point colors identify individual
model families:

| Point color | Model |
|---|---|
| blue | Llama-3.1 8B |
| orange | Mistral 7B |
| green | Gemma-2 9B |
| purple | OLMo-2 7B |
| brown | Qwen2.5 14B |

The current sign agreement is:

| Objective | Contrast | Replication |
|---|---|---:|
| lexical binding | early > late on acquisition | 4/5 models |
| factual association | late > early on transfer | 5/5 models |
| behavioral policy | middle > late on boundedness | 5/5 models |
| causal mapping | middle > early on transfer | 4/5 models |
| procedural reasoning | middle > early on transfer | 5/5 models |

This is the clearest positive robustness result: several targeted localization
predictions replicate by sign across most or all model families.

## Figure Descriptions for the Paper

### Figure: Full-Stack Performance at Llama-Selected Budgets

File:

```text
<ARCHIVE_ROOT>/figures/cross_model_full_stack_performance.{svg,png,pdf}
```

This figure checks whether each model can learn each objective at the
Llama-selected budget before interpreting localization geometry. Rows are
objectives, columns are models, and panels are acquisition, transfer, and
boundedness. Cell values are mean scores across seeds. Dark outlines mark
model-objective pairs flagged by the calibration-mismatch screen.

Recommended caption:

```text
Full-stack adaptation performance at budgets selected from the original
Llama-3.1-8B calibration. Cells show mean performance across three seeds for
each model-objective pair. Acquisition is the mean of ID and paraphrase strict
accuracy, transfer is generalization strict accuracy, and boundedness is the
objective-specific negative-control metric. Outlined cells indicate
model-objective pairs where the Llama-selected budget appears mismatched by the
predefined screening rule.
```

### Figure: Localized-Minus-Full Geometry

File:

```text
<ARCHIVE_ROOT>/figures/cross_model_delta_geometry_heatmap.{svg,png,pdf}
```

This is the main adaptation-geometry figure. It shows whether early, middle, and
late adaptation deviate from full-stack adaptation in the same direction across
models and seeds.

Recommended caption:

```text
Cross-model localized-minus-full adaptation geometry. Each cell reports the
mean paired Delta between a localized LoRA window and full-stack adaptation for
the same model, objective, seed, and metric. Values are percentage points; +/-
denotes a t-based 95% confidence interval across model-seed pairs. Asterisks
mark intervals excluding zero. The figure summarizes whether localized
adaptation produces systematic acquisition, transfer, or boundedness tradeoffs
relative to full-stack adaptation.
```

### Figure: Best Localized Region Agreement

File:

```text
<ARCHIVE_ROOT>/figures/cross_model_best_region_agreement.{svg,png,pdf}
```

This figure is useful when the paper asks whether the "best" region is stable
across architectures.

Recommended caption:

```text
Agreement across model families on the best localized adaptation window. For
each objective and metric, the best localized condition is selected within each
model after averaging across seeds. Cells show the modal best localized region
and the number of models agreeing with that region. Darker cells indicate higher
cross-model agreement.
```

### Figure: Primary Mislocation Penalties

File:

```text
<ARCHIVE_ROOT>/figures/cross_model_primary_mislocation.{svg,png,pdf}
```

This is the strongest robustness summary because it tests prespecified
directional contrasts.

Recommended caption:

```text
Cross-model replication of primary mislocation penalties. For each objective,
the plotted contrast is the prespecified preferred localized region minus a
mislocated region on the primary diagnostic metric. Points are model-level
deltas averaged across seeds, bars show the mean model-level delta, and
horizontal intervals show 95% confidence intervals across models. Point color
identifies the model family, as shown in the legend. The sign-test p-value tests
whether the preferred region outperforms the mislocated region in more models
than expected by chance.
```

### Figure: Geometry Similarity and Variance Components

File:

```text
<ARCHIVE_ROOT>/figures/cross_model_similarity_variance.{svg,png,pdf}
```

This figure summarizes global stability of the 9-dimensional geometry profiles.

Recommended caption:

```text
Similarity and variance decomposition of cross-model adaptation-geometry
profiles. Pairwise cosine similarity is computed between model-mean
localized-minus-full geometry profiles for each objective. Bars show mean +/-
SD across model pairs. The variance panel reports descriptive sums-of-squares
ratios for objective, model, and residual components in the same geometry
space, along with label-permutation p-values.
```

## Calibration-Mismatch Interpretation

Because the budgets were selected using Llama, the first question is whether
full-stack adaptation at that budget is a fair anchor for each model-objective
pair. If full-stack adaptation is already severely underperforming for a model,
then early/middle/late comparisons may reflect budget mismatch rather than only
localization geometry.

The diagnostic table is:

```text
<ARCHIVE_ROOT>/figures/cross_model_calibration_mismatch_diagnostics.csv
```

The screening rule flags a model-objective pair as high priority if any of the
following hold:

- full-stack acquisition < 0.60
- full-stack transfer < 0.35
- full-stack acquisition is at least 0.25 below Llama on the same objective
- full-stack transfer is at least 0.35 below Llama on the same objective

It flags a pair as "consider recalibration" if any of the following softer
conditions hold:

- full-stack acquisition < 0.75
- full-stack transfer < 0.50
- full-stack acquisition is at least 0.15 below Llama
- full-stack transfer is at least 0.20 below Llama
- boundedness < 0.35

This is a triage rule, not a hypothesis test.

### High-Priority Recalibration Candidates

The current high-priority model-objective pairs are:

| Model | Objective | Reason |
|---|---|---|
| Gemma-2 9B | lexical binding | full transfer is low and about 0.40 below Llama |
| Gemma-2 9B | factual association | full acquisition is low and about 0.35 below Llama |
| Gemma-2 9B | causal mapping | full acquisition and transfer are both low, and boundedness is also poor |

These should be treated carefully in the paper. For these cells, the
Llama-selected budget appears substantially mismatched, so localization
geometry may be confounded by under-adaptation or model-specific training
dynamics.

### Secondary Recalibration Candidates

The current "consider recalibration" candidates are:

| Model | Objective | Reason |
|---|---|---|
| Gemma-2 9B | procedural reasoning | full acquisition and transfer are below the desired range and below Llama |
| Qwen2.5 14B | factual association | acquisition is modestly below Llama |
| Llama-3.1 8B | factual association | boundedness is below 0.35 |

The Llama factual-association boundedness flag is not a cross-model budget
mismatch, because Llama is the calibration model. It is better interpreted as a
known acquisition/boundedness tradeoff in the original selected-budget setting.

## Should We Run Another Experiment?

Yes, but it should be targeted rather than a full rerun.

Recommended next experiment:

1. Run model-specific full-stack budget sweeps for the high-priority pairs:
   - Gemma-2 9B / lexical binding
   - Gemma-2 9B / factual association
   - Gemma-2 9B / causal mapping
2. Include Gemma-2 9B / procedural reasoning as a secondary candidate if compute
   allows.
3. Use the same datasets, evaluators, and scoring logic.
4. Sweep budgets around and above the Llama-selected value. For example:
   - lexical binding: 6, 10, 14, 20
   - factual association: 8, 12, 16, 24
   - causal mapping: 10, 16, 24, 32
   - procedural reasoning: 8, 12, 16, 24
5. Select a model-objective budget only after checking all three metrics:
   acquisition, transfer, and boundedness.
6. Re-run localized adaptation at the selected model-specific budget only for
   pairs where the budget sweep materially changes full-stack performance.

A practical selection rule would be:

```text
choose the smallest budget with acquisition >= 0.75,
transfer not severely degraded,
and boundedness not worse than the neighboring larger budget.
```

For objectives where transfer is the main scientific target, require transfer
to improve relative to the Llama-selected budget before using the new budget in
the localization robustness analysis.

## Paper-Safe Interpretation

The strongest current claim is:

```text
Several prespecified localization contrasts replicate in sign across most or
all model families, especially factual association late > early on transfer,
behavioral policy middle > late on boundedness, and procedural reasoning middle
> early on transfer.
```

The claim that should be avoided or softened is:

```text
Objective identity explains more adaptation-geometry variation than model
family.
```

The current variance decomposition does not support that stronger statement:
the model component is larger than the objective component in this run.

Recommended paper wording:

```text
Cross-model results show that adaptation geometry is not purely idiosyncratic
to Llama-3.1-8B: prespecified mislocation penalties replicate across model
families for several objectives. At the same time, model family explains a
substantial share of geometry variance, and some model-objective pairs show
evidence that Llama-selected budgets are mismatched. We therefore interpret the
cross-model experiment as evidence for partial robustness of localized
adaptation signatures, with targeted model-specific calibration needed for the
most mismatched cells.
```

## Regeneration Commands

Rebuild the top-level summaries from all result subdirectories:

```bash
python3 scripts/experiments/localization/run_cross_model_localization.py \
  --summary_only \
  --output_root <ARCHIVE_ROOT>/results/cross_model_localization
```

Rebuild geometry analysis:

```bash
python3 scripts/reproduce/analyze_cross_model_geometry.py \
  --input artifacts/localization/cross_model/summary_cross_model_by_seed.csv \
  --output_dir artifacts/localization/cross_model/geometry_analysis
```

Rebuild figures and calibration diagnostics:

```bash
python3 scripts/reproduce/make_cross_model_figures.py \
  --summary artifacts/localization/cross_model/summary_cross_model_by_seed.csv \
  --geometry_dir artifacts/localization/cross_model/geometry_analysis \
  --output_dir <ARCHIVE_ROOT>/results/cross_model_localization/figures \
  --formats svg png pdf
```

PNG/PDF export uses Chrome or Chromium in headless mode. If the script cannot
find Chrome automatically on a cluster machine, pass `--chrome_exe
/path/to/chrome`. To regenerate only the source vector figures, pass
`--formats svg`.
