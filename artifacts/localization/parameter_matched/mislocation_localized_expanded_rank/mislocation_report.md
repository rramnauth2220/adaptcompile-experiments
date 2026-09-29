# Mislocation Penalty Analysis

Mislocation penalty is defined as preferred condition minus mislocated condition. Positive values indicate that respecting the adaptation geometry improves performance.

## Strongest positive penalties

- **factual_association / transfer / factual_late_vs_early**: Δ=0.436, 95% CI [0.387, 0.460], preferred=late, mislocated=early, n=3
- **factual_association / balanced_min / factual_late_vs_early**: Δ=0.312, 95% CI [0.261, 0.353], preferred=late, mislocated=early, n=3
- **factual_association / transfer / factual_late_vs_middle**: Δ=0.271, 95% CI [0.233, 0.307], preferred=late, mislocated=middle, n=3
- **factual_association / balanced_mean / factual_late_vs_early**: Δ=0.258, 95% CI [0.238, 0.271], preferred=late, mislocated=early, n=3
- **lexical_binding / boundedness / lexical_early_vs_late**: Δ=0.244, 95% CI [0.180, 0.360], preferred=early, mislocated=late, n=3
- **behavioral_policy / boundedness / behavioral_gating_middle_vs_late**: Δ=0.238, 95% CI [0.167, 0.313], preferred=middle, mislocated=late, n=3
- **behavioral_policy / balanced_min / behavioral_gating_middle_vs_late**: Δ=0.228, 95% CI [0.145, 0.313], preferred=middle, mislocated=late, n=3
- **procedural_reasoning / acquisition / procedural_middle_vs_late**: Δ=0.224, 95% CI [0.172, 0.268], preferred=middle, mislocated=late, n=3
- **factual_association / boundedness / factual_late_vs_early**: Δ=0.191, 95% CI [0.073, 0.287], preferred=late, mislocated=early, n=3
- **lexical_binding / acquisition / lexical_early_vs_late**: Δ=0.183, 95% CI [0.136, 0.208], preferred=early, mislocated=late, n=3
- **lexical_binding / boundedness / lexical_early_vs_middle**: Δ=0.176, 95% CI [0.133, 0.200], preferred=early, mislocated=middle, n=3
- **procedural_reasoning / transfer / procedural_middle_vs_early**: Δ=0.164, 95% CI [0.133, 0.193], preferred=middle, mislocated=early, n=3
- **factual_association / balanced_mean / factual_late_vs_middle**: Δ=0.164, 95% CI [0.116, 0.198], preferred=late, mislocated=middle, n=3
- **factual_association / balanced_min / factual_late_vs_middle**: Δ=0.156, 95% CI [0.133, 0.200], preferred=late, mislocated=middle, n=3
- **causal_mapping / balanced_min / causal_middle_vs_early**: Δ=0.156, 95% CI [0.133, 0.167], preferred=middle, mislocated=early, n=3
- **causal_mapping / boundedness / causal_middle_vs_early**: Δ=0.156, 95% CI [0.133, 0.167], preferred=middle, mislocated=early, n=3
- **factual_association / acquisition / factual_late_vs_early**: Δ=0.148, 95% CI [0.124, 0.180], preferred=late, mislocated=early, n=3
- **lexical_binding / balanced_min / lexical_early_vs_late**: Δ=0.136, 95% CI [0.060, 0.193], preferred=early, mislocated=late, n=3
- **lexical_binding / balanced_mean / lexical_early_vs_late**: Δ=0.135, 95% CI [0.101, 0.163], preferred=early, mislocated=late, n=3
- **factual_association / acquisition / factual_late_vs_middle**: Δ=0.125, 95% CI [0.100, 0.140], preferred=late, mislocated=middle, n=3

## Contrast-level summary

### behavioral_acquisition_late_vs_middle

- Objective: `behavioral_policy`
- Component: policy acquisition
- Preferred: `late`
- Mislocated: `middle`
- Interpretation: middle-layer adaptation is less suited than late-layer adaptation for action-label acquisition
- Acquisition penalty: 0.031 [0.008, 0.076]
- Transfer penalty: -0.073 [-0.087, -0.053]
- Boundedness penalty: -0.238 [-0.313, -0.167]
- Balanced mean penalty: -0.093 [-0.131, -0.070]

### behavioral_gating_middle_vs_late

- Objective: `behavioral_policy`
- Component: policy gating / boundedness
- Preferred: `middle`
- Mislocated: `late`
- Interpretation: late-layer adaptation preserves policy actions but is mislocated for bounded policy application
- Acquisition penalty: -0.031 [-0.076, -0.008]
- Transfer penalty: 0.073 [0.053, 0.087]
- Boundedness penalty: 0.238 [0.167, 0.313]
- Balanced mean penalty: 0.093 [0.070, 0.131]

### causal_middle_vs_early

- Objective: `causal_mapping`
- Component: causal transfer
- Preferred: `middle`
- Mislocated: `early`
- Interpretation: early-layer adaptation is mislocated for robust causal transfer
- Acquisition penalty: -0.008 [-0.012, -0.004]
- Transfer penalty: 0.027 [-0.047, 0.087]
- Boundedness penalty: 0.156 [0.133, 0.167]
- Balanced mean penalty: 0.058 [0.036, 0.082]

### causal_middle_vs_late

- Objective: `causal_mapping`
- Component: causal transfer
- Preferred: `middle`
- Mislocated: `late`
- Interpretation: late-layer adaptation is less suited than middle-layer adaptation for causal transfer
- Acquisition penalty: 0.048 [0.028, 0.068]
- Transfer penalty: 0.029 [-0.093, 0.133]
- Boundedness penalty: 0.053 [0.040, 0.060]
- Balanced mean penalty: 0.043 [-0.002, 0.080]

### factual_late_vs_early

- Objective: `factual_association`
- Component: relational fact learning
- Preferred: `late`
- Mislocated: `early`
- Interpretation: early-layer adaptation is mislocated for factual association
- Acquisition penalty: 0.148 [0.124, 0.180]
- Transfer penalty: 0.436 [0.387, 0.460]
- Boundedness penalty: 0.191 [0.073, 0.287]
- Balanced mean penalty: 0.258 [0.238, 0.271]

### factual_late_vs_middle

- Objective: `factual_association`
- Component: relational fact learning
- Preferred: `late`
- Mislocated: `middle`
- Interpretation: middle-layer adaptation is less suited than late-layer adaptation for factual association
- Acquisition penalty: 0.125 [0.100, 0.140]
- Transfer penalty: 0.271 [0.233, 0.307]
- Boundedness penalty: 0.096 [-0.020, 0.160]
- Balanced mean penalty: 0.164 [0.116, 0.198]

### lexical_early_vs_late

- Objective: `lexical_binding`
- Component: lexical binding
- Preferred: `early`
- Mislocated: `late`
- Interpretation: late-layer adaptation is mislocated for lexical acquisition and bounded lexical binding
- Acquisition penalty: 0.183 [0.136, 0.208]
- Transfer penalty: -0.022 [-0.080, 0.087]
- Boundedness penalty: 0.244 [0.180, 0.360]
- Balanced mean penalty: 0.135 [0.101, 0.163]

### lexical_early_vs_middle

- Objective: `lexical_binding`
- Component: lexical binding
- Preferred: `early`
- Mislocated: `middle`
- Interpretation: middle-layer adaptation is less suited than early-layer adaptation for lexical binding
- Acquisition penalty: 0.081 [0.052, 0.108]
- Transfer penalty: -0.087 [-0.473, 0.160]
- Boundedness penalty: 0.176 [0.133, 0.200]
- Balanced mean penalty: 0.057 [-0.063, 0.134]

### procedural_middle_vs_early

- Objective: `procedural_reasoning`
- Component: procedural transfer
- Preferred: `middle`
- Mislocated: `early`
- Interpretation: early-layer adaptation is mislocated for procedural transfer
- Acquisition penalty: 0.060 [-0.040, 0.156]
- Transfer penalty: 0.164 [0.133, 0.193]
- Boundedness penalty: 0.002 [-0.413, 0.213]
- Balanced mean penalty: 0.076 [-0.052, 0.165]

### procedural_middle_vs_late

- Objective: `procedural_reasoning`
- Component: balanced procedural application
- Preferred: `middle`
- Mislocated: `late`
- Interpretation: late-layer adaptation is more conservative and less suited for balanced procedural application
- Acquisition penalty: 0.224 [0.172, 0.268]
- Transfer penalty: 0.100 [-0.033, 0.200]
- Boundedness penalty: -0.129 [-0.300, 0.013]
- Balanced mean penalty: 0.065 [0.056, 0.071]

