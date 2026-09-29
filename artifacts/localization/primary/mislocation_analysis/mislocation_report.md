# Mislocation Penalty Analysis

Mislocation penalty is defined as preferred condition minus mislocated condition. Positive values indicate that respecting the adaptation geometry improves performance.

## Strongest positive penalties

- **factual_association / transfer / factual_late_vs_early**: Δ=0.427, 95% CI [0.353, 0.473], preferred=late, mislocated=early, n=3
- **lexical_binding / boundedness / lexical_early_vs_late**: Δ=0.343, 95% CI [0.293, 0.427], preferred=early, mislocated=late, n=3
- **procedural_reasoning / boundedness / procedural_middle_vs_early**: Δ=0.324, 95% CI [0.273, 0.393], preferred=middle, mislocated=early, n=3
- **factual_association / balanced_min / factual_late_vs_early**: Δ=0.317, 95% CI [0.243, 0.379], preferred=late, mislocated=early, n=3
- **procedural_reasoning / transfer / procedural_middle_vs_early**: Δ=0.287, 95% CI [0.233, 0.340], preferred=middle, mislocated=early, n=3
- **factual_association / transfer / factual_late_vs_middle**: Δ=0.278, 95% CI [0.247, 0.293], preferred=late, mislocated=middle, n=3
- **procedural_reasoning / balanced_min / procedural_middle_vs_early**: Δ=0.276, 95% CI [0.233, 0.307], preferred=middle, mislocated=early, n=3
- **factual_association / balanced_mean / factual_late_vs_early**: Δ=0.273, 95% CI [0.234, 0.305], preferred=late, mislocated=early, n=3
- **lexical_binding / boundedness / lexical_early_vs_middle**: Δ=0.238, 95% CI [0.207, 0.280], preferred=early, mislocated=middle, n=3
- **behavioral_policy / boundedness / behavioral_gating_middle_vs_late**: Δ=0.216, 95% CI [0.153, 0.267], preferred=middle, mislocated=late, n=3
- **causal_mapping / boundedness / causal_middle_vs_early**: Δ=0.216, 95% CI [0.167, 0.313], preferred=middle, mislocated=early, n=3
- **causal_mapping / balanced_min / causal_middle_vs_early**: Δ=0.216, 95% CI [0.167, 0.313], preferred=middle, mislocated=early, n=3
- **lexical_binding / acquisition / lexical_early_vs_late**: Δ=0.209, 95% CI [0.172, 0.236], preferred=early, mislocated=late, n=3
- **procedural_reasoning / balanced_mean / procedural_middle_vs_early**: Δ=0.206, 95% CI [0.175, 0.252], preferred=middle, mislocated=early, n=3
- **factual_association / acquisition / factual_late_vs_early**: Δ=0.197, 95% CI [0.180, 0.216], preferred=late, mislocated=early, n=3
- **factual_association / boundedness / factual_late_vs_early**: Δ=0.196, 95% CI [0.033, 0.307], preferred=late, mislocated=early, n=3
- **factual_association / balanced_min / factual_late_vs_middle**: Δ=0.175, 95% CI [0.156, 0.199], preferred=late, mislocated=middle, n=3
- **factual_association / balanced_mean / factual_late_vs_middle**: Δ=0.174, 95% CI [0.132, 0.199], preferred=late, mislocated=middle, n=3
- **behavioral_policy / balanced_min / behavioral_gating_middle_vs_late**: Δ=0.159, 95% CI [0.096, 0.227], preferred=middle, mislocated=late, n=3
- **procedural_reasoning / acquisition / procedural_middle_vs_late**: Δ=0.156, 95% CI [0.120, 0.212], preferred=middle, mislocated=late, n=3

## Contrast-level summary

### behavioral_acquisition_late_vs_middle

- Objective: `behavioral_policy`
- Component: policy acquisition
- Preferred: `late`
- Mislocated: `middle`
- Interpretation: middle-layer adaptation is less suited than late-layer adaptation for action-label acquisition
- Acquisition penalty: 0.051 [0.008, 0.104]
- Transfer penalty: -0.029 [-0.080, 0.053]
- Boundedness penalty: -0.216 [-0.267, -0.153]
- Balanced mean penalty: -0.065 [-0.082, -0.036]

### behavioral_gating_middle_vs_late

- Objective: `behavioral_policy`
- Component: policy gating / boundedness
- Preferred: `middle`
- Mislocated: `late`
- Interpretation: late-layer adaptation preserves policy actions but is mislocated for bounded policy application
- Acquisition penalty: -0.051 [-0.104, -0.008]
- Transfer penalty: 0.029 [-0.053, 0.080]
- Boundedness penalty: 0.216 [0.153, 0.267]
- Balanced mean penalty: 0.065 [0.036, 0.082]

### causal_middle_vs_early

- Objective: `causal_mapping`
- Component: causal transfer
- Preferred: `middle`
- Mislocated: `early`
- Interpretation: early-layer adaptation is mislocated for robust causal transfer
- Acquisition penalty: 0.051 [-0.012, 0.152]
- Transfer penalty: 0.129 [0.020, 0.220]
- Boundedness penalty: 0.216 [0.167, 0.313]
- Balanced mean penalty: 0.132 [0.066, 0.204]

### causal_middle_vs_late

- Objective: `causal_mapping`
- Component: causal transfer
- Preferred: `middle`
- Mislocated: `late`
- Interpretation: late-layer adaptation is less suited than middle-layer adaptation for causal transfer
- Acquisition penalty: 0.023 [-0.004, 0.052]
- Transfer penalty: 0.093 [0.087, 0.100]
- Boundedness penalty: 0.053 [0.040, 0.060]
- Balanced mean penalty: 0.056 [0.052, 0.060]

### factual_late_vs_early

- Objective: `factual_association`
- Component: relational fact learning
- Preferred: `late`
- Mislocated: `early`
- Interpretation: early-layer adaptation is mislocated for factual association
- Acquisition penalty: 0.197 [0.180, 0.216]
- Transfer penalty: 0.427 [0.353, 0.473]
- Boundedness penalty: 0.196 [0.033, 0.307]
- Balanced mean penalty: 0.273 [0.234, 0.305]

### factual_late_vs_middle

- Objective: `factual_association`
- Component: relational fact learning
- Preferred: `late`
- Mislocated: `middle`
- Interpretation: middle-layer adaptation is less suited than late-layer adaptation for factual association
- Acquisition penalty: 0.141 [0.112, 0.156]
- Transfer penalty: 0.278 [0.247, 0.293]
- Boundedness penalty: 0.102 [-0.007, 0.167]
- Balanced mean penalty: 0.174 [0.132, 0.199]

### lexical_early_vs_late

- Objective: `lexical_binding`
- Component: lexical binding
- Preferred: `early`
- Mislocated: `late`
- Interpretation: late-layer adaptation is mislocated for lexical acquisition and bounded lexical binding
- Acquisition penalty: 0.209 [0.172, 0.236]
- Transfer penalty: -0.144 [-0.193, -0.053]
- Boundedness penalty: 0.343 [0.293, 0.427]
- Balanced mean penalty: 0.136 [0.112, 0.159]

### lexical_early_vs_middle

- Objective: `lexical_binding`
- Component: lexical binding
- Preferred: `early`
- Mislocated: `middle`
- Interpretation: middle-layer adaptation is less suited than early-layer adaptation for lexical binding
- Acquisition penalty: 0.108 [0.076, 0.132]
- Transfer penalty: -0.101 [-0.243, 0.067]
- Boundedness penalty: 0.238 [0.207, 0.280]
- Balanced mean penalty: 0.082 [0.026, 0.123]

### procedural_middle_vs_early

- Objective: `procedural_reasoning`
- Component: procedural transfer
- Preferred: `middle`
- Mislocated: `early`
- Interpretation: early-layer adaptation is mislocated for procedural transfer
- Acquisition penalty: 0.007 [-0.040, 0.076]
- Transfer penalty: 0.287 [0.233, 0.340]
- Boundedness penalty: 0.324 [0.273, 0.393]
- Balanced mean penalty: 0.206 [0.175, 0.252]

### procedural_middle_vs_late

- Objective: `procedural_reasoning`
- Component: balanced procedural application
- Preferred: `middle`
- Mislocated: `late`
- Interpretation: late-layer adaptation is more conservative and less suited for balanced procedural application
- Acquisition penalty: 0.156 [0.120, 0.212]
- Transfer penalty: 0.038 [-0.007, 0.073]
- Boundedness penalty: -0.056 [-0.180, 0.013]
- Balanced mean penalty: 0.046 [-0.004, 0.095]

