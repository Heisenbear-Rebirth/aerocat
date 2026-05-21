# C2 — Policy-Level Conservatism Mechanism (post-A1/A2)

Trajectory data from 5 seeds × 5 λ × 100 episodes per group on T1 OOD eval.

Active-step-weighted distributions; all values are 5-seed mean ± SD.


## Table C2-I. Per-metric plateau at λ=1.0 (5-seed mean ± SD)

| Group | |action|₂ mean | |action|₂ p95 | env-std mean | thrust mean | sat trig | v_err mean |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| A | 0.408 ± 0.057 | 0.725 ± 0.038 | 0.232 ± 0.015 | -0.314 ± 0.086 | 0.320 ± 0.063 | 5.274 ± 0.066 |
| B | 0.253 ± 0.031 | 0.454 ± 0.038 | 0.189 ± 0.024 | -0.137 ± 0.039 | 0.292 ± 0.075 | 5.963 ± 0.083 |
| C | 0.391 ± 0.023 | 0.716 ± 0.025 | 0.227 ± 0.010 | -0.294 ± 0.026 | 0.308 ± 0.038 | 5.276 ± 0.056 |
| D | 0.260 ± 0.027 | 0.486 ± 0.058 | 0.209 ± 0.017 | -0.084 ± 0.036 | 0.282 ± 0.055 | 5.896 ± 0.083 |
| E | 0.255 ± 0.025 | 0.497 ± 0.039 | 0.205 ± 0.011 | -0.089 ± 0.043 | 0.290 ± 0.071 | 5.863 ± 0.052 |
| F | 0.424 ± 0.035 | 0.742 ± 0.026 | 0.236 ± 0.014 | -0.325 ± 0.060 | 0.337 ± 0.024 | 5.209 ± 0.069 |

## Table C2-II. Paired contrasts on |action|₂ mean @ λ=1.0

Positive Δ ⇒ variant has higher action magnitude.

| Contrast | Type | Δ |action|₂ | t (df=4) | Cohen d | p<0.05? |
|:--:|:--:|:--:|:--:|:--:|:--:|
| D - A | cross-reward (main C1 anchor) | -0.1481 | -5.97 | -2.67 | ✓ |
| D - C | cross-reward | -0.1309 | -8.04 | -3.60 | ✓ |
| D - F | cross-reward (vs Cai) | -0.1639 | -13.29 | -5.94 | ✓ |
| B - A | cross-reward (within-MLP) | -0.1555 | -4.32 | -1.93 | ✓ |
| E - A | cross-reward | -0.1538 | -8.23 | -3.68 | ✓ |
| C - A | within-dense | -0.0172 | -0.66 | -0.29 | n.s. |
| F - A | within-dense | +0.0158 | +0.75 | +0.33 | n.s. |
| C - F | within-dense | -0.0330 | -1.97 | -0.88 | n.s. |
| D - B | within-sparse | +0.0074 | +0.34 | +0.15 | n.s. |
| E - B | within-sparse | +0.0017 | +0.07 | +0.03 | n.s. |
| D - E | within-sparse | +0.0057 | +0.45 | +0.20 | n.s. |

## Table C2-III/IV. Paired contrasts on per-env action std @ λ=1.0

| Contrast | Type | Δ env_std_mean | t (df=4) | Cohen d | p<0.05? |
|:--:|:--:|:--:|:--:|:--:|:--:|
| D - A | cross-reward (main C1 anchor) | -0.0228 | -1.92 | -0.86 | n.s. |
| D - C | cross-reward | -0.0185 | -1.56 | -0.70 | n.s. |
| D - F | cross-reward (vs Cai) | -0.0270 | -2.46 | -1.10 | n.s. |
| B - A | cross-reward (within-MLP) | -0.0431 | -3.91 | -1.75 | ✓ |
| E - A | cross-reward | -0.0267 | -2.81 | -1.26 | ✓ |
| C - A | within-dense | -0.0043 | -0.85 | -0.38 | n.s. |
| F - A | within-dense | +0.0042 | +0.59 | +0.27 | n.s. |
| C - F | within-dense | -0.0085 | -1.72 | -0.77 | n.s. |
| D - B | within-sparse | +0.0203 | +1.71 | +0.76 | n.s. |
| E - B | within-sparse | +0.0163 | +1.12 | +0.50 | n.s. |
| D - E | within-sparse | +0.0040 | +0.59 | +0.27 | n.s. |

## Table C2-III/IV. Paired contrasts on saturation triggering rate @ λ=1.0

| Contrast | Type | Δ sat_trig_rate | t (df=4) | Cohen d | p<0.05? |
|:--:|:--:|:--:|:--:|:--:|:--:|
| D - A | cross-reward (main C1 anchor) | -0.0373 | -0.86 | -0.38 | n.s. |
| D - C | cross-reward | -0.0260 | -0.83 | -0.37 | n.s. |
| D - F | cross-reward (vs Cai) | -0.0541 | -1.67 | -0.75 | n.s. |
| B - A | cross-reward (within-MLP) | -0.0277 | -0.81 | -0.36 | n.s. |
| E - A | cross-reward | -0.0299 | -0.59 | -0.26 | n.s. |
| C - A | within-dense | -0.0113 | -0.82 | -0.37 | n.s. |
| F - A | within-dense | +0.0167 | +0.64 | +0.29 | n.s. |
| C - F | within-dense | -0.0280 | -1.44 | -0.64 | n.s. |
| D - B | within-sparse | -0.0096 | -0.20 | -0.09 | n.s. |
| E - B | within-sparse | -0.0022 | -0.04 | -0.02 | n.s. |
| D - E | within-sparse | -0.0074 | -0.61 | -0.27 | n.s. |

## Table C2-V. |action|₂ mean vs λ (5-seed mean)

| Group | λ=0.0 | λ=0.3 | λ=0.5 | λ=0.7 | λ=1.0 |
|:-:|:-:|:-:|:-:|:-:|:-:|
| A | 0.770 | 0.656 | 0.559 | 0.478 | 0.408 |
| B | 0.413 | 0.335 | 0.293 | 0.273 | 0.253 |
| C | 0.755 | 0.630 | 0.535 | 0.459 | 0.391 |
| D | 0.544 | 0.387 | 0.313 | 0.279 | 0.260 |
| E | 0.554 | 0.400 | 0.319 | 0.281 | 0.255 |
| F | 0.787 | 0.678 | 0.581 | 0.499 | 0.424 |