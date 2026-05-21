# C3 — Stochastic-Policy OOD Safety Validation

Deterministic baseline = `_dr_eval_results.json` (P0).  Stochastic = `_c3/T1_stoch_results.json`.

Action: deterministic `tanh(mean)` vs stochastic `tanh(mean+std*eps)`.


## Table C3-I. Crash rate @ λ=1.0 (5-seed mean ± SD)

| Group | deterministic | stochastic | Δ (stoch−det) |
|:-:|:-:|:-:|:-:|
| A | 0.358 ± 0.179 | 0.394 ± 0.242 | +0.036 |
| B | 0.178 ± 0.036 | 0.146 ± 0.015 | -0.032 |
| C | 0.492 ± 0.035 | 0.518 ± 0.180 | +0.026 |
| D | 0.150 ± 0.029 | 0.150 ± 0.032 | -0.000 |
| E | 0.172 ± 0.011 | 0.168 ± 0.034 | -0.004 |
| F | 0.408 ± 0.160 | 0.528 ± 0.213 | +0.120 |

## Table C3-II. Sparse-vs-dense crash gap @ λ=1.0: det vs stoch

Gap = dense − sparse (positive = sparse safer). Retention = stoch_gap / det_gap.

| Contrast | det gap | stoch gap | retention | det t | stoch t |
|:-:|:-:|:-:|:-:|:-:|:-:|
| A−D | +0.208 (*) | +0.244 (n.s.) | 1.17× | +2.56 | +2.08 |
| A−B | +0.180 (n.s.) | +0.248 (*) | 1.38× | +1.97 | +2.31 |
| A−E | +0.186 (*) | +0.226 (*) | 1.22× | +2.24 | +2.18 |
| C−D | +0.342 (***) | +0.368 (**) | 1.08× | +12.62 | +4.41 |
| F−D | +0.258 (**) | +0.378 (**) | 1.47× | +4.08 | +3.78 |
| C−B | +0.314 (***) | +0.372 (***) | 1.18× | +12.65 | +4.72 |
| F−B | +0.230 (**) | +0.382 (**) | 1.66× | +3.40 | +4.20 |

## Table C3-III. 'All sparse < all dense' ordering @ λ=1.0

| Policy | max sparse (B/D/E) | min dense (A/C/F) | clean separation? |
|:-:|:-:|:-:|:-:|
| deterministic | 0.178 | 0.358 | YES |
| stochastic | 0.168 | 0.394 | YES |

## Table C3-IV. Crash rate vs λ — A & D, det vs stoch (5-seed mean)

| λ | A det | A stoch | D det | D stoch |
|:-:|:-:|:-:|:-:|:-:|
| 0.0 | 0.000 | 0.000 | 0.000 | 0.000 |
| 0.3 | 0.000 | 0.000 | 0.000 | 0.000 |
| 0.5 | 0.000 | 0.000 | 0.000 | 0.000 |
| 0.7 | 0.034 | 0.036 | 0.018 | 0.010 |
| 1.0 | 0.358 | 0.394 | 0.150 | 0.150 |