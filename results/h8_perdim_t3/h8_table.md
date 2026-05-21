# H5 — Per-Dimension OOD Perturbation Decomposition

Eval: 6 groups × 5 seeds × 9 dims, deterministic policy, 100 episodes per cell, 256 envs.

Dims activate λ=1 in one perturbation axis; all others stay λ=0.

Sanity: nominal ≈ baseline (λ=0); all_l1 ≈ P0 λ=1.0 results.


## (1) Crash rate per (group, dim), mean ± SD over 5 seeds

| Group | nominal | mass | wind | turb | sensor | actuator | init_state | collision | all_l1 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| A | 0.0±0.0 | 0.0±0.0 | 0.0±0.0 | 0.0±0.0 | 0.0±0.0 | 0.0±0.0 | 0.0±0.0 | 39.2±2.9 | 19.0±2.0 |
| C | 0.0±0.0 | 0.0±0.0 | 0.0±0.0 | 0.0±0.0 | 0.0±0.0 | 0.0±0.0 | 0.0±0.0 | 37.8±3.4 | 24.0±10.0 |
| F | 0.0±0.0 | 0.0±0.0 | 0.0±0.0 | 0.0±0.0 | 0.0±0.0 | 0.0±0.0 | 0.0±0.0 | 39.0±5.1 | 19.8±2.6 |
| D | 0.0±0.0 | 0.0±0.0 | 0.0±0.0 | 0.0±0.0 | 0.0±0.0 | 0.0±0.0 | 0.0±0.0 | 19.6±1.8 | 19.6±3.9 |

(Values in percent. SD across 5 seeds.)


## (2) Δcrash = dim − nominal per (group, dim) (paired by seed)

How much does activating ONE OOD dim alone add to crash rate vs full-nominal?

| Group | mass | wind | turb | sensor | actuator | init_state | collision | all_l1 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| A | +0.0** | +0.0** | +0.0** | +0.0** | +0.0** | +0.0** | +39.2** | +19.0** |
| C | +0.0** | +0.0** | +0.0** | +0.0** | +0.0** | +0.0** | +37.8** | +24.0** |
| F | +0.0** | +0.0** | +0.0** | +0.0** | +0.0** | +0.0** | +39.0** | +19.8** |
| D | +0.0** | +0.0** | +0.0** | +0.0** | +0.0** | +0.0** | +19.6** | +19.6** |

** = p<0.05 (|t|>2.776, df=4); · = p<0.10. Percent units.


## (3) Sanity check: nominal vs all_l1 (no P0 cross-ref here; manually compare)

| Group | nominal crash | all_l1 crash | Δ |
|:---:|:---:|:---:|:---:|
| A | 0.0% | 19.0% | +19.0% |
| C | 0.0% | 24.0% | +24.0% |
| F | 0.0% | 19.8% | +19.8% |
| D | 0.0% | 19.6% | +19.6% |

## (4) Per-group ranking — which dim adds the most crash?

| Group | Top-1 dim (Δcrash) | Top-2 | Top-3 |
|:---:|:---:|:---:|:---:|
| A | collision (+39.2%) | mass (+0.0%) | wind (+0.0%) |
| C | collision (+37.8%) | mass (+0.0%) | wind (+0.0%) |
| F | collision (+39.0%) | mass (+0.0%) | wind (+0.0%) |
| D | collision (+19.6%) | mass (+0.0%) | wind (+0.0%) |

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
| collision | -19.6% | -16.81 | 5 | ↓ D safer |
| all_l1 | +0.6% | +0.24 | 5 | n.s. |

## (6) Tracking RMSE per (group, dim), mean over 5 seeds

Captures non-crash failure modes (e.g. mass changes hurt tracking but not safety).

| Group | nominal | mass | wind | turb | sensor | actuator | init_state | collision | all_l1 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| A | 2.42 | 2.46 | 3.64 | 2.42 | 2.42 | 2.21 | 3.58 | 3.31 | 7.63 |
| C | 2.37 | 2.39 | 3.58 | 2.37 | 2.35 | 2.20 | 3.59 | 3.41 | 6.39 |
| F | 2.39 | 2.40 | 3.50 | 2.39 | 2.39 | 2.21 | 3.57 | 3.23 | 6.31 |
| D | 3.94 | 3.42 | 5.14 | 3.94 | 3.94 | 3.56 | 4.15 | 4.10 | 9.00 |

## (7) Mean episode length (steps) per (group, dim)

CRITICAL CAVEAT: per-episode crash rate confounds with exposure time. Dense (A/C/F)

episodes run to ~500 steps (timeout); sparse (B/D/E) terminate at goal in 5-50 steps.

Section (8) reports per-second crash hazard for fair safety comparison.

| Group | nominal | mass | wind | turb | sensor | actuator | init_state | collision | all_l1 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| A | 500 | 2 | 372 | 500 | 500 | 500 | 490 | 317 | 12 |
| C | 500 | 2 | 381 | 500 | 500 | 500 | 490 | 367 | 40 |
| F | 500 | 2 | 346 | 500 | 500 | 500 | 489 | 357 | 13 |
| D | 374 | 4 | 134 | 374 | 371 | 406 | 341 | 11 | 7 |

## (8) Per-second crash hazard rate (1/s) per (group, dim), mean over 5 seeds

hazard = crash_rate / (mean_ep_length × 0.02s).

Removes exposure-time confound: 'given 1 second of OOD flight, P(crash)'.

Use for fair safety comparison across groups with different episode termination patterns.

| Group | nominal | mass | wind | turb | sensor | actuator | init_state | collision | all_l1 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| A | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.067 | 0.855 |
| C | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.054 | 0.731 |
| F | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.060 | 0.918 |
| D | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.002 | 1.484 |

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
| collision | +0.936 | +5.15 | ↑ A per-step safer |
| all_l1 | +0.629 | +3.57 | ↑ A per-step safer |