# Cross-Model Geometry Report

- Models: 5 (gemma_2_9b_it, llama_3_1_8b_instruct, mistral_7b_instruct_v0_3, olmo_2_1124_7b_instruct, qwen2_5_14b_instruct)
- Objectives: 5 (behavioral_policy, causal_mapping, factual_association, lexical_binding, procedural_reasoning)
- Seeds: 3 (11, 22, 33)

## Best Region Agreement

- behavioral_policy / acquisition: late (3/5)
- behavioral_policy / transfer: late (3/5)
- behavioral_policy / boundedness: middle (5/5)
- causal_mapping / acquisition: middle (3/5)
- causal_mapping / transfer: middle (4/5)
- causal_mapping / boundedness: middle (2/5)
- factual_association / acquisition: late (3/5)
- factual_association / transfer: late (4/5)
- factual_association / boundedness: middle (2/5)
- lexical_binding / acquisition: early (4/5)
- lexical_binding / transfer: early (3/5)
- lexical_binding / boundedness: early (3/5)
- procedural_reasoning / acquisition: late (2/5)
- procedural_reasoning / transfer: late (3/5)
- procedural_reasoning / boundedness: late (4/5)

## Geometry Similarity

- behavioral_policy: mean cosine=0.778885003044132, mean Euclidean=0.7517482922433776
- causal_mapping: mean cosine=0.735968183899386, mean Euclidean=0.46741513545999375
- factual_association: mean cosine=0.8420878666805457, mean Euclidean=0.6294037319856184
- lexical_binding: mean cosine=0.6899907110804987, mean Euclidean=0.859986100646094
- procedural_reasoning: mean cosine=0.587780886591264, mean Euclidean=0.7461472294079993

## Variance Decomposition

- Objective explained ratio: 0.25009279850113253
- Model explained ratio: 0.34205118470658813
- Larger descriptive component: model

## Permutation Tests

- Objective-label permutation p: 9.999000099990002e-05
- Model-label permutation p: 9.999000099990002e-05

## Primary Mislocation Sign Agreement

- lexical_binding / acquisition: 4/5 positive for early > late
- factual_association / transfer: 5/5 positive for late > early
- behavioral_policy / boundedness: 5/5 positive for middle > late
- causal_mapping / transfer: 4/5 positive for middle > early
- procedural_reasoning / transfer: 5/5 positive for middle > early

## Interpretation Notes

Best-region agreement and geometry similarity summarize which localization signatures replicate across model families. In this descriptive analysis, the model variance component is larger than the objective component, so the current partial run does not by itself support the claim that task type explains more geometry variation than architecture family.
