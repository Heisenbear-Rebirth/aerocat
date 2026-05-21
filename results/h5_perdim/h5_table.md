# H5 — Per-Dimension OOD Perturbation Decomposition

Eval: 6 groups × 5 seeds × 9 dims, deterministic policy, 100 episodes per cell, 256 envs.

Dims activate λ=1 in one perturbation axis; all others stay λ=0.

Sanity: nominal ≈ baseline (λ=0); all_l1 ≈ P0 λ=1.0 results.


## (1) Crash rate per (group, dim), mean ± SD over 5 seeds

| Group | nominal | mass | wind | turb | sensor | actuator | init_state | collision | all_l1 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| A | 0.0±0.0 | 0.0±0.0 | 0.0±0.0 | 0.0±0.0 | 0.0±0.0 | 0.0±0.0 | 0.0±0.0 | 40.2±7.3 | 20.0±1.6 |
| B | 0.0±0.0 | 0.0±0.0 | 0.0±0.0 | 0.0±0.0 | 0.0±0.0 | 0.0±0.0 | 0.0±0.0 | 18.2±1.5 | 20.8±4.4 |
| C | 0.0±0.0 | 0.0±0.0 | 0.0±0.0 | 0.0±0.0 | 0.0±0.0 | 0.0±0.0 | 0.0±0.0 | 40.4±2.3 | 19.6±3.7 |
| D | 0.0±0.0 | 0.0±0.0 | 0.0±0.0 | 0.0±0.0 | 0.0±0.0 | 0.0±0.0 | 0.0±0.0 | 19.6±3.0 | 19.4±2.6 |
| E | 0.0±0.0 | 0.0±0.0 | 0.0±0.0 | 0.0±0.0 | 0.0±0.0 | 0.0±0.0 | 0.0±0.0 | 19.0±2.7 | 20.2±2.6 |
| F | 0.0±0.0 | 0.0±0.0 | 0.0±0.0 | 0.0±0.0 | 0.0±0.0 | 0.0±0.0 | 0.0±0.0 | 38.8±3.7 | 23.8±3.3 |

(Values in percent. SD across 5 seeds.)


## (2) Δcrash = dim − nominal per (group, dim) (paired by seed)

How much does activating ONE OOD dim alone add to crash rate vs full-nominal?

| Group | mass | wind | turb | sensor | actuator | init_state | collision | all_l1 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| A | +0.0** | +0.0** | +0.0** | +0.0** | +0.0** | +0.0** | +40.2** | +20.0** |
| B | +0.0** | +0.0** | +0.0** | +0.0** | +0.0** | +0.0** | +18.2** | +20.8** |
| C | +0.0** | +0.0** | +0.0** | +0.0** | +0.0** | +0.0** | +40.4** | +19.6** |
| D | +0.0** | +0.0** | +0.0** | +0.0** | +0.0** | +0.0** | +19.6** | +19.4** |
| E | +0.0** | +0.0** | +0.0** | +0.0** | +0.0** | +0.0** | +19.0** | +20.2** |
| F | +0.0** | +0.0** | +0.0** | +0.0** | +0.0** | +0.0** | +38.8** | +23.8** |

** = p<0.05 (|t|>2.776, df=4); · = p<0.10. Percent units.


## (3) Sanity check: nominal vs all_l1 (no P0 cross-ref here; manually compare)

| Group | nominal crash | all_l1 crash | Δ |
|:---:|:---:|:---:|:---:|
| A | 0.0% | 20.0% | +20.0% |
| B | 0.0% | 20.8% | +20.8% |
| C | 0.0% | 19.6% | +19.6% |
| D | 0.0% | 19.4% | +19.4% |
| E | 0.0% | 20.2% | +20.2% |
| F | 0.0% | 23.8% | +23.8% |

## (4) Per-group ranking — which dim adds the most crash?

| Group | Top-1 dim (Δcrash) | Top-2 | Top-3 |
|:---:|:---:|:---:|:---:|
| A | collision (+40.2%) | mass (+0.0%) | wind (+0.0%) |
| B | collision (+18.2%) | mass (+0.0%) | wind (+0.0%) |
| C | collision (+40.4%) | mass (+0.0%) | wind (+0.0%) |
| D | collision (+19.6%) | mass (+0.0%) | wind (+0.0%) |
| E | collision (+19.0%) | mass (+0.0%) | wind (+0.0%) |
| F | collision (+38.8%) | mass (+0.0%) | wind (+0.0%) |

## (5) D − A crash gap per dim (paired by seed)

Tests whether PSC+sparse's OOD-safety advantage over MLP+dense holds dim-by-dim.

Negative ⇒ D safer than A in that dim.

| Dim | D − A Δcrash | t (df=4) | n | Direction |
|:---:|:---:|:---:|:---:|:---:|
| nominal | +0.0% | +inf | 5 | ↑ A safer |
| mass | +0.0% | +inf | 5 | ↑ A safer |
| wind | +0.0% | +inf | 5 | ↑ A safer |
| turb | +0.0% | +inf | 5 | ↑ A safer |
| sensor | +0.0% | +inf | 5 | ↑ A safer |
| actuator | +0.0% | +inf | 5 | ↑ A safer |
| init_state | +0.0% | +inf | 5 | ↑ A safer |
| collision | -20.6% | -8.98 | 5 | ↓ D safer |
| all_l1 | -0.6% | -0.43 | 5 | n.s. |

## (6) Tracking RMSE per (group, dim), mean over 5 seeds

Captures non-crash failure modes (e.g. mass changes hurt tracking but not safety).

| Group | nominal | mass | wind | turb | sensor | actuator | init_state | collision | all_l1 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| A | 2.38 | 2.43 | 3.58 | 2.39 | 2.39 | 2.22 | 3.68 | 3.29 | 6.38 |
| B | 4.72 | 3.67 | 5.59 | 4.72 | 4.72 | 4.26 | 4.42 | 4.40 | 8.95 |
| C | 2.39 | 2.44 | 3.58 | 2.40 | 2.38 | 2.23 | 3.57 | 3.34 | 6.51 |
| D | 3.85 | 3.45 | 5.01 | 3.85 | 3.85 | 3.41 | 4.13 | 4.00 | 11.36 |
| E | 3.89 | 3.42 | 5.16 | 3.89 | 3.89 | 3.41 | 4.15 | 4.09 | 10.19 |
| F | 2.38 | 2.35 | 3.57 | 2.38 | 2.39 | 2.21 | 3.60 | 3.19 | 6.32 |

## (7) Mean episode length (steps) per (group, dim)

CRITICAL CAVEAT: per-episode crash rate confounds with exposure time. Dense (A/C/F)

episodes run to ~500 steps (timeout); sparse (B/D/E) terminate at goal in 5-50 steps.

Section (8) reports per-second crash hazard for fair safety comparison.

| Group | nominal | mass | wind | turb | sensor | actuator | init_state | collision | all_l1 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| A | 500 | 2 | 369 | 500 | 500 | 500 | 489 | 329 | 10 |
| B | 226 | 4 | 70 | 226 | 233 | 404 | 207 | 5 | 6 |
| C | 500 | 2 | 412 | 500 | 500 | 500 | 490 | 315 | 15 |
| D | 358 | 4 | 130 | 357 | 351 | 410 | 348 | 10 | 8 |
| E | 378 | 4 | 79 | 378 | 361 | 416 | 358 | 8 | 7 |
| F | 500 | 2 | 369 | 500 | 500 | 500 | 489 | 342 | 11 |

## (8) Per-second crash hazard rate (1/s) per (group, dim), mean over 5 seeds

hazard = crash_rate / (mean_ep_length × 0.02s).

Removes exposure-time confound: 'given 1 second of OOD flight, P(crash)'.

Use for fair safety comparison across groups with different episode termination patterns.

| Group | nominal | mass | wind | turb | sensor | actuator | init_state | collision | all_l1 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| A | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.102 | 1.032 |
| B | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.786 | 1.881 |
| C | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.066 | 0.707 |
| D | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.076 | 1.237 |
| E | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.230 | 1.562 |
| F | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.060 | 1.156 |

## (9) D − A hazard rate gap per dim (paired by seed)

Repeats (5) but using per-second hazard. Negative ⇒ D's per-step robustness > A's.

| Dim | D − A Δhazard (/s) | t (df=4) | Direction |
|:---:|:---:|:---:|:---:|
| nominal | +0.000 | +inf | ↑ A per-step safer |
| mass | +0.000 | +inf | ↑ A per-step safer |
| wind | +0.000 | +inf | ↑ A per-step safer |
| turb | +0.000 | +inf | ↑ A per-step safer |
| sensor | +0.000 | +inf | ↑ A per-step safer |
| actuator | +0.000 | +inf | ↑ A per-step safer |
| init_state | +0.000 | +inf | ↑ A per-step safer |
| collision | +0.974 | +5.74 | ↑ A per-step safer |
| all_l1 | +0.205 | +1.18 | n.s. |