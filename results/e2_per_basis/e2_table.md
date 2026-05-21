# E2 — Per-Basis Leave-One-Out Necessity (5 seeds x 1B steps)

Baseline = full PSC group D (all 5 bases). Each row removes one basis.

Metric: last-10% success_rate plateau (same as paper SS V-A).


**Baseline D (full PSC):** per-seed SR plateau = 0.3356, 0.3438, 0.3284, 0.3776, 0.2405

**Baseline D mean +/- SD:** 0.3252 +/- 0.0510


## Table E2-I. Per-basis ablation vs full-D baseline

| Removed basis | Abl SR (mean +/- SD) | Delta vs D | rel drop % | t (df=4) | Cohen d | verdict |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|
| phi0 vel_err | 0.3628 +/- 0.0598 | +0.0376 | +11.6% | +2.33 | +1.04 | * HARMFUL(!) |
| phi1 omega | 0.2977 +/- 0.0911 | -0.0275 | -8.5% | -0.74 | -0.33 | n.s. redundant |
| phi2 tilt | 0.3422 +/- 0.1024 | +0.0170 | +5.2% | +0.59 | +0.26 | n.s. redundant |
| phi3 PID_integral | 0.3683 +/- 0.0246 | +0.0432 | +13.3% | +1.44 | +0.64 | n.s. redundant |
| phi4 saturation | 0.2931 +/- 0.0657 | -0.0321 | -9.9% | -0.86 | -0.38 | n.s. redundant |

## Table E2-II. Necessity ranking (most necessary = largest SR drop when removed)

| Rank | Basis | Delta SR when removed | verdict |
|:--:|:--|:--:|:--:|
| 1 | phi4 saturation | -0.0321 | redundant |
| 2 | phi1 omega | -0.0275 | redundant |
| 3 | phi2 tilt | +0.0170 | redundant |
| 4 | phi0 vel_err | +0.0376 | HARMFUL(!) |
| 5 | phi3 PID_integral | +0.0432 | redundant |

## Table E2-III. Raw per-seed SR plateaus

| Config | seed 42 | seed 123 | seed 456 | seed 789 | seed 1024 | mean |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|
| D (full) | 0.3356 | 0.3438 | 0.3284 | 0.3776 | 0.2405 | 0.3252 |
| noPhi0 (phi0 vel_err) | 0.4316 | 0.3818 | 0.3576 | 0.3749 | 0.2678 | 0.3628 |
| noPhi1 (phi1 omega) | 0.2919 | 0.2815 | 0.4440 | 0.2790 | 0.1919 | 0.2977 |
| noPhi2 (phi2 tilt) | 0.3863 | 0.4427 | 0.2710 | 0.4106 | 0.2004 | 0.3422 |
| noPhi3 (phi3 PID_integral) | 0.3720 | 0.3957 | 0.3444 | 0.3419 | 0.3879 | 0.3683 |
| noPhi4 (phi4 saturation) | 0.3985 | 0.2274 | 0.2502 | 0.2952 | 0.2941 | 0.2931 |