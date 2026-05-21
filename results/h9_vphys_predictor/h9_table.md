# H9 — V_phys as deployment-time crash predictor (AUROC, K=20 steps)

Reports AUROC ≥ 0.5 (after auto-direction flip; direction tag indicates predictive sign).

Random guess = 0.5. AUROC > 0.7 = useful predictor; > 0.8 = strong.

Crash label uses reward < 0 at done step as crash criterion (heuristic).

Positive-class rate per cell ranges 0.1–5%; AUROC handles class imbalance.


## Predicting: imminent ANY termination within K=20 steps

### AUROC mean ± SD across 5 seeds, per (group, λ)


#### Predictor: V_phys

| Group | λ=0.0 | λ=0.3 | λ=0.5 | λ=0.7 | λ=1.0 |
|:---:|:---:|:---:|:---:|:---:|:---:|
| C | 0.539±0.010 (low) | 0.511±0.009 (high) | 0.520±0.015 (high) | 0.633±0.022 (high) | 0.668±0.037 (high) |
| D | 0.537±0.059 (low) | 0.627±0.089 (high) | 0.708±0.081 (high) | 0.736±0.058 (high) | 0.754±0.034 (high) |
| E | 0.534±0.008 (low) | 0.553±0.046 (high) | 0.664±0.045 (high) | 0.709±0.063 (high) | 0.738±0.058 (high) |

#### Predictor: V_res

| Group | λ=0.0 | λ=0.3 | λ=0.5 | λ=0.7 | λ=1.0 |
|:---:|:---:|:---:|:---:|:---:|:---:|
| C | 0.528±0.026 (low) | 0.532±0.009 (low) | 0.515±0.011 (low) | 0.645±0.018 (high) | 0.684±0.032 (high) |
| D | 0.518±0.018 (low) | 0.592±0.061 (high) | 0.658±0.057 (high) | 0.673±0.035 (high) | 0.690±0.033 (high) |
| E | 0.506±0.004 (high) | 0.549±0.033 (high) | 0.635±0.037 (high) | 0.666±0.066 (high) | 0.701±0.032 (high) |

#### Predictor: V

| Group | λ=0.0 | λ=0.3 | λ=0.5 | λ=0.7 | λ=1.0 |
|:---:|:---:|:---:|:---:|:---:|:---:|
| A | 0.525±0.015 (low) | 0.537±0.012 (low) | 0.528±0.018 (low) | 0.622±0.020 (high) | 0.708±0.029 (high) |
| B | 0.525±0.029 (high) | 0.623±0.055 (high) | 0.642±0.020 (high) | 0.641±0.049 (high) | 0.624±0.077 (high) |
| C | 0.529±0.026 (low) | 0.531±0.009 (low) | 0.514±0.012 (low) | 0.647±0.019 (high) | 0.685±0.033 (high) |
| D | 0.515±0.021 (low) | 0.593±0.062 (high) | 0.661±0.059 (high) | 0.679±0.036 (high) | 0.698±0.030 (high) |
| E | 0.515±0.005 (low) | 0.549±0.036 (high) | 0.639±0.036 (high) | 0.673±0.066 (high) | 0.710±0.032 (high) |
| F | 0.538±0.024 (low) | 0.541±0.012 (low) | 0.516±0.015 (low) | 0.601±0.055 (high) | 0.709±0.056 (high) |

### Paired Δ AUROC = AUROC[V_phys] − AUROC[V_total] within PSC groups (5 seeds, label=imm)

Positive ⇒ V_phys alone is a better/comparable predictor than full V (interpretable substitute).

Negative ⇒ V_total beats V_phys (V_phys adds noise relative to V_res).

| Group | λ | Mean Δ | t (df=4) | n | Direction |
|:---:|:---:|:---:|:---:|:---:|:---:|
| C | 0.0 | +0.010 | +1.09 | 5 | n.s. |
| C | 0.3 | -0.020 | -2.57 | 5 | trend (p<0.10) |
| C | 0.5 | +0.007 | +0.64 | 5 | n.s. |
| C | 0.7 | -0.014 | -2.20 | 5 | trend (p<0.10) |
| C | 1.0 | -0.017 | -3.45 | 5 | ↓ V_phys < V |
| D | 0.0 | +0.021 | +0.85 | 5 | n.s. |
| D | 0.3 | +0.035 | +2.57 | 5 | trend (p<0.10) |
| D | 0.5 | +0.046 | +3.35 | 5 | ↑ V_phys ≥ V |
| D | 0.7 | +0.056 | +4.46 | 5 | ↑ V_phys ≥ V |
| D | 1.0 | +0.056 | +2.86 | 5 | ↑ V_phys ≥ V |
| E | 0.0 | +0.019 | +5.84 | 5 | ↑ V_phys ≥ V |
| E | 0.3 | +0.004 | +0.68 | 5 | n.s. |
| E | 0.5 | +0.024 | +1.54 | 5 | n.s. |
| E | 0.7 | +0.036 | +2.00 | 5 | n.s. |
| E | 1.0 | +0.029 | +0.98 | 5 | n.s. |

## Predicting: imminent CRASH (reward<0 at done) within K=20 steps

### AUROC mean ± SD across 5 seeds, per (group, λ)


#### Predictor: V_phys

| Group | λ=0.0 | λ=0.3 | λ=0.5 | λ=0.7 | λ=1.0 |
|:---:|:---:|:---:|:---:|:---:|:---:|
| C | 0.982±0.004 (low) | 0.973±0.005 (low) | 0.957±0.010 (low) | 0.888±0.039 (low) | 0.810±0.018 (low) |
| D | 0.887±0.108 (low) | 0.634±0.068 (low) | 0.691±0.132 (low) | 0.751±0.053 (high) | 0.833±0.064 (high) |
| E | 0.948±0.017 (low) | 0.806±0.034 (low) | 0.750±0.008 (low) | 0.713±0.161 (high) | 0.796±0.029 (high) |

#### Predictor: V_res

| Group | λ=0.0 | λ=0.3 | λ=0.5 | λ=0.7 | λ=1.0 |
|:---:|:---:|:---:|:---:|:---:|:---:|
| C | 0.977±0.004 (low) | 0.973±0.007 (low) | 0.954±0.007 (low) | 0.878±0.029 (low) | 0.774±0.039 (low) |
| D | 0.925±0.145 (low) | 0.942±0.017 (low) | 0.893±0.015 (low) | 0.686±0.148 (low) | 0.699±0.075 (high) |
| E | 0.995±0.001 (low) | 0.939±0.016 (low) | 0.898±0.033 (low) | 0.682±0.109 (high) | 0.719±0.045 (high) |

#### Predictor: V

| Group | λ=0.0 | λ=0.3 | λ=0.5 | λ=0.7 | λ=1.0 |
|:---:|:---:|:---:|:---:|:---:|:---:|
| A | 0.981±0.003 (low) | 0.974±0.003 (low) | 0.959±0.011 (low) | 0.915±0.016 (low) | 0.772±0.047 (low) |
| B | — | 0.966±0.035 (low) | 0.925±0.015 (low) | 0.671±0.140 (high) | 0.631±0.056 (high) |
| C | 0.980±0.004 (low) | 0.974±0.005 (low) | 0.957±0.009 (low) | 0.886±0.031 (low) | 0.792±0.031 (low) |
| D | 0.996±0.003 (low) | 0.939±0.015 (low) | 0.890±0.015 (low) | 0.682±0.145 (low) | 0.710±0.071 (high) |
| E | 0.995±0.001 (low) | 0.934±0.015 (low) | 0.895±0.033 (low) | 0.684±0.113 (high) | 0.730±0.042 (high) |
| F | 0.978±0.004 (low) | 0.973±0.004 (low) | 0.958±0.008 (low) | 0.906±0.016 (low) | 0.803±0.027 (low) |

### Paired Δ AUROC = AUROC[V_phys] − AUROC[V_total] within PSC groups (5 seeds, label=crash)

Positive ⇒ V_phys alone is a better/comparable predictor than full V (interpretable substitute).

Negative ⇒ V_total beats V_phys (V_phys adds noise relative to V_res).

| Group | λ | Mean Δ | t (df=4) | n | Direction |
|:---:|:---:|:---:|:---:|:---:|:---:|
| C | 0.0 | +0.002 | +2.29 | 5 | trend (p<0.10) |
| C | 0.3 | -0.002 | -0.74 | 5 | n.s. |
| C | 0.5 | +0.001 | +0.38 | 5 | n.s. |
| C | 0.7 | +0.002 | +0.52 | 5 | n.s. |
| C | 1.0 | +0.018 | +2.04 | 5 | n.s. |
| D | 0.0 | -0.109 | -2.00 | 4 | n.s. |
| D | 0.3 | -0.306 | -11.35 | 5 | ↓ V_phys < V |
| D | 0.5 | -0.199 | -3.11 | 4 | ↓ V_phys < V |
| D | 0.7 | +0.069 | +1.04 | 5 | n.s. |
| D | 1.0 | +0.123 | +2.74 | 5 | trend (p<0.10) |
| E | 0.0 | -0.046 | -4.17 | 2 | ↓ V_phys < V |
| E | 0.3 | -0.128 | -8.19 | 5 | ↓ V_phys < V |
| E | 0.5 | -0.145 | -10.99 | 5 | ↓ V_phys < V |
| E | 0.7 | +0.030 | +0.77 | 5 | n.s. |
| E | 1.0 | +0.067 | +4.72 | 5 | ↑ V_phys ≥ V |

## Highlights: AUROC at λ=1.0 (OOD-extreme, K=20)

Single-number summary for the harshest OOD condition.

| Group | Predictor | label_imm AUROC (mean±SD) | label_crash AUROC (mean±SD) | n_seeds |
|:---:|:---:|:---:|:---:|:---:|
| C | V_phys | 0.668±0.037 (high) | 0.810±0.018 (low) | 5 |
| C | V_res | 0.684±0.032 (high) | 0.774±0.039 (low) | 5 |
| C | V | 0.685±0.033 (high) | 0.792±0.031 (low) | 5 |
| D | V_phys | 0.754±0.034 (high) | 0.833±0.064 (high) | 5 |
| D | V_res | 0.690±0.033 (high) | 0.699±0.075 (high) | 5 |
| D | V | 0.698±0.030 (high) | 0.710±0.071 (high) | 5 |
| E | V_phys | 0.738±0.058 (high) | 0.796±0.029 (high) | 5 |
| E | V_res | 0.701±0.032 (high) | 0.719±0.045 (high) | 5 |
| E | V | 0.710±0.032 (high) | 0.730±0.042 (high) | 5 |