# H1 — Deployment-time $|V_{\mathrm{phys}}|/(|V_{\mathrm{phys}}|+|V_{\mathrm{res}}|)$ vs OOD $\lambda$

Tests whether PSC's V_phys takes over as V_res fails under OOD.

Data: C1 T1_calib_trajectories, 6 groups × 5 seeds × 5 lambdas = 150 cells.

Ratio uses mean(|V_phys|) and mean(|V_res|) over active+valid steps.


## (1) Mean ratio per (group, lambda), 5 seeds

MLP-only groups (A/B/F) have V_phys=0 by construction → ratio=0 (omitted).

| Group | λ=0.0 | λ=0.3 | λ=0.5 | λ=0.7 | λ=1.0 |
|:---:|:---:|:---:|:---:|:---:|:---:|
| C | 0.212 ± 0.003 | 0.243 ± 0.003 | 0.275 ± 0.002 | 0.310 ± 0.003 | 0.367 ± 0.003 |
| D | 0.153 ± 0.019 | 0.186 ± 0.022 | 0.199 ± 0.027 | 0.220 ± 0.025 | 0.179 ± 0.023 |
| E | 0.128 ± 0.007 | 0.153 ± 0.009 | 0.175 ± 0.016 | 0.171 ± 0.025 | 0.141 ± 0.023 |

## (2) Paired Δratio = ratio[λ=1.0] − ratio[λ=0.0] (5 seeds)

Positive ⇒ V_phys takes over under OOD (V_res fails). Negative ⇒ V_res still dominates (or grows).

| Group | Mean Δratio | t (df=4) | Cohen's d | n | Direction |
|:---:|:---:|:---:|:---:|:---:|:---:|
| C | +0.155 | +95.06 | +42.51 | 5 | ↑ V_phys takes over |
| D | +0.026 | +1.87 | +0.84 | 5 | n.s. |
| E | +0.012 | +0.97 | +0.43 | 5 | n.s. |

## (3) Component magnitudes (mean ± std over 5 seeds)

If ratio rises with λ, decompose: is it because |V_phys| rises, |V_res| falls, or both?


### Group C

| Quantity | λ=0.0 | λ=0.3 | λ=0.5 | λ=0.7 | λ=1.0 |
|:---:|:---:|:---:|:---:|:---:|:---:|
| |V_phys| | 19.99±0.03 | 19.31±0.08 | 18.32±0.20 | 16.90±0.20 | 15.00±0.24 |
| |V_res| | 74.14±1.19 | 60.10±0.77 | 48.40±0.97 | 37.57±0.89 | 25.83±0.67 |
| |V| | 93.92±1.15 | 79.07±0.69 | 66.31±1.22 | 53.88±1.15 | 39.89±0.92 |
| |V−G| | 27.57±0.18 | 25.12±0.15 | 24.05±0.38 | 23.06±0.51 | 23.51±0.74 |

### Group D

| Quantity | λ=0.0 | λ=0.3 | λ=0.5 | λ=0.7 | λ=1.0 |
|:---:|:---:|:---:|:---:|:---:|:---:|
| |V_phys| | 20.03±0.64 | 19.73±0.45 | 19.81±1.23 | 19.82±0.90 | 17.73±0.54 |
| |V_res| | 112.79±16.71 | 87.80±12.89 | 81.97±18.84 | 71.12±8.43 | 82.48±12.27 |
| |V| | 130.97±17.23 | 102.24±13.22 | 91.99±21.91 | 77.34±7.57 | 79.03±9.04 |
| |V−G| | 233.94±43.48 | 120.49±17.49 | 122.83±13.24 | 98.78±38.23 | 160.78±65.86 |

### Group E

| Quantity | λ=0.0 | λ=0.3 | λ=0.5 | λ=0.7 | λ=1.0 |
|:---:|:---:|:---:|:---:|:---:|:---:|
| |V_phys| | 17.43±0.28 | 16.58±0.18 | 15.51±0.23 | 15.21±1.23 | 12.87±1.09 |
| |V_res| | 118.87±9.10 | 92.15±7.21 | 73.57±8.77 | 75.06±14.36 | 79.97±11.18 |
| |V| | 134.57±9.13 | 105.13±7.36 | 81.48±9.13 | 79.81±15.42 | 80.78±11.82 |
| |V−G| | 246.20±26.41 | 116.99±3.88 | 119.41±13.31 | 128.28±30.17 | 116.40±22.66 |

## (4) Calibration error |V−G| across λ (all groups, mean only)

If PSC stays calibrated while MLP degrades, that's deployment-time V calibration evidence.

| Group | λ=0.0 | λ=0.3 | λ=0.5 | λ=0.7 | λ=1.0 |
|:---:|:---:|:---:|:---:|:---:|:---:|
| A | 27.27 | 25.17 | 23.77 | 22.86 | 31.60 |
| B | 161.64 | 77.42 | 93.58 | 75.94 | 102.69 |
| C | 27.57 | 25.12 | 24.05 | 23.06 | 23.51 |
| D | 233.94 | 120.49 | 122.83 | 98.78 | 160.78 |
| E | 246.20 | 116.99 | 119.41 | 128.28 | 116.40 |
| F | 27.21 | 26.12 | 25.22 | 23.77 | 27.32 |

## (5) Paired Δ|V−G| = err[λ=1.0] − err[λ=0.0] per group (5 seeds)

Negative = calibration **improves** under OOD (counter-intuitive). Positive = degrades (expected).

| Group | Mean Δ|V−G| | t (df=4) | d | n | Direction |
|:---:|:---:|:---:|:---:|:---:|:---:|
| A | +4.33 | +0.90 | +0.40 | 5 | n.s. |
| B | -58.95 | -1.99 | -0.89 | 5 | n.s. |
| C | -4.06 | -10.89 | -4.87 | 5 | ↓ improves |
| D | -73.15 | -2.31 | -1.03 | 5 | ↓ (p<0.10) |
| E | -129.80 | -5.96 | -2.66 | 5 | ↓ improves |
| F | +0.11 | +0.02 | +0.01 | 5 | n.s. |

## (6) Dense-reward calibration: paired contrast on Δ|V−G| (A vs C vs F)

Tests whether PSC (C) keeps |V−G| lower under OOD than MLP (A) or Cai dual (F).

Each row pairs by seed and compares one group's Δ|V−G| to another's.

| Contrast | Mean diff | t (df=4) | d | n | Direction |
|:---:|:---:|:---:|:---:|:---:|:---:|
| C - A | -8.39 | -1.85 | -0.83 | 5 | n.s. |
| F - A | -4.22 | -1.76 | -0.79 | 5 | n.s. |
| C - F | -4.17 | -0.99 | -0.44 | 5 | n.s. |