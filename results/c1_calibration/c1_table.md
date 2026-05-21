# C1 — OOD V Calibration Analysis (deterministic policy)

Observed return G reconstructed per-env via backward sweep (gamma=0.995), masking trajectory tails that lacked a within-buffer episode termination.

Metrics on full V_theta(s) = V_phys + V_res; auxiliary metrics on V_phys and V_res alone (PSC groups only).


## Table C1-I. V_theta calibration @ lambda=1.0 (5-seed mean)

| Group | n_valid | corr(V,G) | bias | RMSE | explained_var | G_mean | G_std |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| A | 77091 | +0.555 | +21.34 | 39.39 | -0.005 | +20.48 | 36.42 |
| B | 1023 | +0.308 | +34.81 | 132.74 | -2.836 | -52.55 | 95.48 |
| C | 127479 | +0.748 | +10.31 | 30.70 | +0.559 | +23.62 | 43.54 |
| D | 1347 | +0.377 | +118.25 | 239.00 | -1.930 | -127.64 | 207.09 |
| E | 1169 | +0.222 | +59.49 | 153.06 | -9.479 | -55.10 | 114.50 |
| F | 102180 | +0.755 | +9.41 | 34.39 | -1.111 | +19.29 | 37.29 |

## Table C1-II. PSC vs MLP V-calibration paired contrasts @ lambda=1.0

Positive Delta corr = PSC's V better correlated with G; negative Delta RMSE = PSC's V closer to G; Delta EV: positive = PSC's V explains more variance.

| Contrast | type | Delta corr | t | Delta RMSE | t | Delta EV | t |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| C - A | dense | +0.194 (n.s.) | +1.62 | -8.687 (n.s.) | -1.82 | +0.564 (n.s.) | +1.60 |
| F - A | dense | +0.200 (n.s.) | +1.56 | -4.995 (n.s.) | -1.80 | -1.106 (n.s.) | -0.80 |
| C - F | dense | -0.007 (n.s.) | -0.48 | -3.692 (n.s.) | -0.86 | +1.670 (n.s.) | +1.01 |
| D - B | sparse | +0.069 (n.s.) | +1.30 | +106.264 (*) | +2.74 | +0.905 (**) | +3.34 |
| E - B | sparse | -0.086 (n.s.) | -1.71 | +20.321 (n.s.) | +0.64 | -6.644 (n.s.) | -0.67 |
| D - E | sparse | +0.155 (***) | +10.72 | +85.942 (n.s.) | +1.40 | +7.549 (n.s.) | +0.76 |

## Table C1-III. V_phys vs V_res calibration (PSC groups, lambda=1.0)

Decomposition: how well does each component alone correlate with G?

| Group | corr(V_phys, G) | RMSE(V_phys, G) | corr(V_res, G) | RMSE(V_res, G) | corr(V_total, G) | RMSE(V_total, G) |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| C | +0.687 | 38.55 | +0.697 | 32.27 | +0.748 | 30.70 |
| D | +0.109 | 254.35 | +0.377 | 231.44 | +0.377 | 239.00 |
| E | +0.164 | 133.25 | +0.220 | 147.22 | +0.222 | 153.06 |

## Table C1-IV. V_theta corr(V,G) vs lambda (5-seed mean)

| Group | lam=0.0 | lam=0.3 | lam=0.5 | lam=0.7 | lam=1.0 |
|:-:|:-:|:-:|:-:|:-:|:-:|
| A | +0.659 | +0.698 | +0.725 | +0.740 | +0.555 |
| B | +0.804 | +0.613 | +0.543 | +0.365 | +0.308 |
| C | +0.654 | +0.704 | +0.736 | +0.750 | +0.748 |
| D | +0.775 | +0.608 | +0.469 | +0.349 | +0.377 |
| E | +0.766 | +0.623 | +0.522 | +0.426 | +0.222 |
| F | +0.662 | +0.696 | +0.727 | +0.743 | +0.755 |