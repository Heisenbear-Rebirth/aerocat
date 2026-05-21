# E1 — Cross-Task PSC Weight Transfer Analysis

Two T3-sparse configs: baseline (default PSC init) vs transfer (D-T1 converged init).

Reference: D-T1 (T1-sparse, full PSC) and A-T3 (T3-dense, MLP baseline).


## Table E1-I. Per-seed SR plateau

| Config | seed 42 | seed 123 | seed 456 | seed 789 | seed 1024 | mean ± SD |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|
| baseline_T3 | 0.3947 | 0.3192 | 0.3144 | 0.3862 | 0.2630 | 0.3355 ± 0.0549 |
| transfer_T3 | 0.3711 | 0.4892 | 0.3617 | 0.3405 | 0.2545 | 0.3634 ± 0.0841 |
| D_T1_ref | 0.3356 | 0.3438 | 0.3284 | 0.3776 | 0.2405 | 0.3252 ± 0.0510 |
| A_T3_ref | 0.5360 | 0.5328 | 0.5383 | 0.5324 | 0.5300 | 0.5339 ± 0.0033 |

## Q1: T3-sparse precondition (does sparse converge on T3?)

- **T3-sparse (baseline) SR plateau** = 0.3355 ± 0.0549
- T1-sparse (D, full PSC) SR plateau = 0.3252 ± 0.0510
- T3-dense (A) SR plateau = 0.5339 ± 0.0033
- T2-sparse (D, from v19.4 deliverable) = 0.026 ± 0.014 (collapse reference)

**Verdict**: **T3-sparse converges to non-degenerate plateau** (0.336 >> 0.026 T2 collapse). Sparse precondition holds on T3.

## Q2: Cold-start speedup — transfer vs baseline (paired seeds)

| Threshold SR | baseline median steps | transfer median steps | speedup b/t |
|:--:|:--:|:--:|:--:|
| 0.05 | 5M (5/5) | 5M (5/5) | 1.00× |
| 0.10 | 182M (5/5) | 218M (5/5) | 0.83× |
| 0.20 | 512M (5/5) | 541M (5/5) | 0.95× |
| 0.30 | 729M (4/5) | 776M (4/5) | 0.94× |

### Paired t-test on time-to-SR-threshold (transfer − baseline; negative = transfer faster)

| Threshold | Δ (transfer − baseline, M steps) | t (df=4) | Cohen d | sig |
|:--:|:--:|:--:|:--:|:--:|
| 0.05 | +0.0 | +inf | +inf | *** |
| 0.10 | +28.5 | +3.26 | +1.46 | ** |
| 0.20 | +14.6 | +0.69 | +0.31 | n.s. |
| 0.30 | -8.6 | -0.11 | -0.05 | n.s. |

## Q3: Final plateau — transfer vs baseline (paired seeds)

- Δ SR plateau (transfer − baseline) = +0.0279  t=+0.72  d=+0.32  n.s.
- Per-seed Δ: ['-0.0236', '+0.1700', '+0.0473', '-0.0457', '-0.0085']