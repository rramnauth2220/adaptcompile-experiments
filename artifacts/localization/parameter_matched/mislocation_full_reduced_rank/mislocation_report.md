# Mislocation Penalty Analysis

Mislocation penalty is defined as preferred condition minus mislocated condition. Positive values indicate that respecting the adaptation geometry improves performance.

## Strongest positive penalties

- **procedural_reasoning / boundedness / procedural_middle_vs_early**: Δ=0.693, 95% CI [0.647, 0.753], preferred=middle, mislocated=early, n=3
- **procedural_reasoning / balanced_min / procedural_middle_vs_early**: Δ=0.542, 95% CI [0.480, 0.633], preferred=middle, mislocated=early, n=3
- **factual_association / transfer / factual_late_vs_early**: Δ=0.487, 95% CI [0.433, 0.520], preferred=late, mislocated=early, n=3
- **factual_association / balanced_min / factual_late_vs_early**: Δ=0.345, 95% CI [0.289, 0.399], preferred=late, mislocated=early, n=3
- **factual_association / transfer / factual_late_vs_middle**: Δ=0.316, 95% CI [0.287, 0.367], preferred=late, mislocated=middle, n=3
- **procedural_reasoning / balanced_mean / procedural_middle_vs_early**: Δ=0.303, 95% CI [0.244, 0.335], preferred=middle, mislocated=early, n=3
- **factual_association / balanced_mean / factual_late_vs_early**: Δ=0.284, 95% CI [0.224, 0.324], preferred=late, mislocated=early, n=3
- **behavioral_policy / boundedness / behavioral_gating_middle_vs_late**: Δ=0.249, 95% CI [0.187, 0.313], preferred=middle, mislocated=late, n=3
- **procedural_reasoning / transfer / procedural_middle_vs_early**: Δ=0.222, 95% CI [0.133, 0.333], preferred=middle, mislocated=early, n=3
- **behavioral_policy / balanced_min / behavioral_gating_middle_vs_late**: Δ=0.216, 95% CI [0.141, 0.313], preferred=middle, mislocated=late, n=3
- **factual_association / boundedness / factual_late_vs_early**: Δ=0.209, 95% CI [0.153, 0.247], preferred=late, mislocated=early, n=3
- **lexical_binding / acquisition / lexical_early_vs_late**: Δ=0.208, 95% CI [0.196, 0.228], preferred=early, mislocated=late, n=3
- **factual_association / balanced_mean / factual_late_vs_middle**: Δ=0.192, 95% CI [0.173, 0.228], preferred=late, mislocated=middle, n=3
- **procedural_reasoning / acquisition / procedural_middle_vs_late**: Δ=0.177, 95% CI [0.124, 0.224], preferred=middle, mislocated=late, n=3
- **factual_association / balanced_min / factual_late_vs_middle**: Δ=0.174, 95% CI [0.143, 0.207], preferred=late, mislocated=middle, n=3
- **factual_association / acquisition / factual_late_vs_early**: Δ=0.157, 95% CI [0.084, 0.220], preferred=late, mislocated=early, n=3
- **factual_association / boundedness / factual_late_vs_middle**: Δ=0.156, 95% CI [0.140, 0.173], preferred=late, mislocated=middle, n=3
- **lexical_binding / balanced_mean / lexical_early_vs_late**: Δ=0.141, 95% CI [0.068, 0.212], preferred=early, mislocated=late, n=3
- **lexical_binding / boundedness / lexical_early_vs_late**: Δ=0.122, 95% CI [0.080, 0.180], preferred=early, mislocated=late, n=3
- **lexical_binding / balanced_min / lexical_early_vs_late**: Δ=0.118, 95% CI [0.080, 0.180], preferred=early, mislocated=late, n=3

## Contrast-level summary

### behavioral_acquisition_late_vs_middle

- Objective: `behavioral_policy`
- Component: policy acquisition
- Preferred: `late`
- Mislocated: `middle`
- Interpretation: middle-layer adaptation is less suited than late-layer adaptation for action-label acquisition
- Acquisition penalty: 0.045 [0.024, 0.060]
- Transfer penalty: -0.040 [-0.067, -0.007]
- Boundedness penalty: -0.249 [-0.313, -0.187]
- Balanced mean penalty: -0.081 [-0.119, -0.044]

### behavioral_gating_middle_vs_late

- Objective: `behavioral_policy`
- Component: policy gating / boundedness
- Preferred: `middle`
- Mislocated: `late`
- Interpretation: late-layer adaptation preserves policy actions but is mislocated for bounded policy application
- Acquisition penalty: -0.045 [-0.060, -0.024]
- Transfer penalty: 0.040 [0.007, 0.067]
- Boundedness penalty: 0.249 [0.187, 0.313]
- Balanced mean penalty: 0.081 [0.044, 0.119]

### causal_middle_vs_early

- Objective: `causal_mapping`
- Component: causal transfer
- Preferred: `middle`
- Mislocated: `early`
- Interpretation: early-layer adaptation is mislocated for robust causal transfer
- Acquisition penalty: 0.004 [-0.008, 0.012]
- Transfer penalty: 0.027 [0.000, 0.073]
- Boundedness penalty: 0.111 [0.000, 0.167]
- Balanced mean penalty: 0.047 [0.004, 0.077]

### causal_middle_vs_late

- Objective: `causal_mapping`
- Component: causal transfer
- Preferred: `middle`
- Mislocated: `late`
- Interpretation: late-layer adaptation is less suited than middle-layer adaptation for causal transfer
- Acquisition penalty: 0.024 [0.000, 0.060]
- Transfer penalty: 0.040 [-0.007, 0.067]
- Boundedness penalty: 0.027 [-0.007, 0.060]
- Balanced mean penalty: 0.030 [0.018, 0.049]

### factual_late_vs_early

- Objective: `factual_association`
- Component: relational fact learning
- Preferred: `late`
- Mislocated: `early`
- Interpretation: early-layer adaptation is mislocated for factual association
- Acquisition penalty: 0.157 [0.084, 0.220]
- Transfer penalty: 0.487 [0.433, 0.520]
- Boundedness penalty: 0.209 [0.153, 0.247]
- Balanced mean penalty: 0.284 [0.224, 0.324]

### factual_late_vs_middle

- Objective: `factual_association`
- Component: relational fact learning
- Preferred: `late`
- Mislocated: `middle`
- Interpretation: middle-layer adaptation is less suited than late-layer adaptation for factual association
- Acquisition penalty: 0.105 [0.080, 0.144]
- Transfer penalty: 0.316 [0.287, 0.367]
- Boundedness penalty: 0.156 [0.140, 0.173]
- Balanced mean penalty: 0.192 [0.173, 0.228]

### lexical_early_vs_late

- Objective: `lexical_binding`
- Component: lexical binding
- Preferred: `early`
- Mislocated: `late`
- Interpretation: late-layer adaptation is mislocated for lexical acquisition and bounded lexical binding
- Acquisition penalty: 0.208 [0.196, 0.228]
- Transfer penalty: 0.093 [-0.100, 0.327]
- Boundedness penalty: 0.122 [0.080, 0.180]
- Balanced mean penalty: 0.141 [0.068, 0.212]

### lexical_early_vs_middle

- Objective: `lexical_binding`
- Component: lexical binding
- Preferred: `early`
- Mislocated: `middle`
- Interpretation: middle-layer adaptation is less suited than early-layer adaptation for lexical binding
- Acquisition penalty: 0.103 [0.096, 0.112]
- Transfer penalty: 0.098 [-0.007, 0.180]
- Boundedness penalty: 0.036 [-0.053, 0.147]
- Balanced mean penalty: 0.079 [0.013, 0.121]

### procedural_middle_vs_early

- Objective: `procedural_reasoning`
- Component: procedural transfer
- Preferred: `middle`
- Mislocated: `early`
- Interpretation: early-layer adaptation is mislocated for procedural transfer
- Acquisition penalty: -0.005 [-0.080, 0.040]
- Transfer penalty: 0.222 [0.133, 0.333]
- Boundedness penalty: 0.693 [0.647, 0.753]
- Balanced mean penalty: 0.303 [0.244, 0.335]

### procedural_middle_vs_late

- Objective: `procedural_reasoning`
- Component: balanced procedural application
- Preferred: `middle`
- Mislocated: `late`
- Interpretation: late-layer adaptation is more conservative and less suited for balanced procedural application
- Acquisition penalty: 0.177 [0.124, 0.224]
- Transfer penalty: 0.100 [0.040, 0.147]
- Boundedness penalty: -0.020 [-0.040, 0.007]
- Balanced mean penalty: 0.086 [0.041, 0.126]

