# B1+ Cross-Validation: RL spontaneous weight drift vs E2 leave-one-out

Hypothesis: RL drift magnitude per basis should agree with E2 leave-one-out impact.


## Per-basis means (n=5 bases, 5-seed averaged)

| basis | init w | final w mean | drift mean | |drift| mean | E2 SR_change mean | E2 SR_change SD |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|
| phi0 vel_err | +45.00 | +44.099 | -0.901 | 0.901 | +0.0376 | 0.0361 |
| phi1 omega | +2.00 | +2.321 | +0.321 | 0.457 | -0.0275 | 0.0829 |
| phi2 tilt | +2.00 | -0.023 | -2.023 | 2.023 | +0.0170 | 0.0650 |
| phi3 PID_integral | +0.50 | -0.542 | -1.042 | 1.042 | +0.0432 | 0.0670 |
| phi4 saturation | +1.00 | +0.061 | -0.939 | 0.939 | -0.0321 | 0.0838 |

**Correlations (n=5 bases)**: Pearson |drift| vs SR_change = +0.361 (p=0.551); Spearman = +0.300 (p=0.624).

**Correlations (n=25 basis-seed pairs)**: Pearson = +0.184 (p=0.379); Spearman = +0.172 (p=0.410).

## Sign-of-drift commentary

- **phi0 vel_err**: init +45.00 → final +44.099; |drift| = 0.90 — RL barely moved this weight. E2 verdict: ΔSR = +0.0376 ± 0.0361
- **phi1 omega**: init +2.00 → final +2.321; |drift| = 0.46 — RL barely moved this weight. E2 verdict: ΔSR = -0.0275 ± 0.0829
- **phi2 tilt**: init +2.00 → final -0.023; **large |drift| = 2.02** — RL actively reshaped this weight (sign crossed across seeds). E2 verdict: ΔSR = +0.0170 ± 0.0650
- **phi3 PID_integral**: init +0.50 → final -0.542; **large |drift| = 1.04** — RL actively reshaped this weight (sign crossed across seeds). E2 verdict: ΔSR = +0.0432 ± 0.0670
- **phi4 saturation**: init +1.00 → final +0.061; |drift| = 0.94 — RL barely moved this weight (sign crossed across seeds). E2 verdict: ΔSR = -0.0321 ± 0.0838