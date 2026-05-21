# A2 — Critic Cold-Start Anchoring Analysis

Tests whether PSC provides cold-start anchor benefit (replacement hypothesis after A1 disproved variance reduction).


## (1) Early vf_loss decay — $\log_{10}(\mathrm{vf\_loss}[T] / \mathrm{vf\_loss}[5M])$

More-negative = critic learning faster. **Within-reward only** (vf_loss is reward-scale-specific).

| Task | Group | log₁₀ ratio @50M | @100M | @200M |
|:---:|:---:|:---:|:---:|:---:|
| T1 | A | +0.048 ± 0.019 | +0.094 ± 0.021 | +0.223 ± 0.013 |
| T1 | B | +0.091 ± 0.039 | +0.159 ± 0.038 | +0.265 ± 0.050 |
| T1 | C | +0.090 ± 0.163 | -0.083 ± 0.225 | -0.147 ± 0.124 |
| T1 | D | +0.146 ± 0.160 | +0.116 ± 0.087 | +0.250 ± 0.047 |
| T1 | E | +0.111 ± 0.186 | +0.167 ± 0.092 | +0.255 ± 0.048 |
| T1 | F | +0.061 ± 0.013 | +0.109 ± 0.016 | +0.233 ± 0.014 |
| T2 | A | +0.015 ± 0.014 | +0.062 ± 0.008 | +0.127 ± 0.022 |
| T2 | B | -0.261 ± 0.024 | -0.541 ± 0.115 | -0.573 ± 0.147 |
| T2 | C | +0.100 ± 0.286 | +0.206 ± 0.284 | -0.067 ± 0.197 |
| T2 | D | +0.074 ± 0.246 | -0.202 ± 0.442 | -0.276 ± 0.124 |
| T2 | F | -0.001 ± 0.009 | +0.053 ± 0.009 | +0.133 ± 0.009 |
| T3 | A | +0.045 ± 0.018 | +0.090 ± 0.020 | +0.216 ± 0.021 |
| T3 | C | +0.055 ± 0.153 | -0.117 ± 0.226 | -0.085 ± 0.075 |
| T3 | F | +0.053 ± 0.016 | +0.107 ± 0.022 | +0.225 ± 0.017 |

### Within-reward paired contrasts on log-ratio @50M

(negative Δ ⇒ PSC variant decays vf_loss faster than baseline)

| Task | Contrast | Δ log-ratio @50M | t (df=4) | d | Direction |
|:---:|:---:|:---:|:---:|:---:|:---:|
| T1 | C - A | +0.043 | +0.57 | +0.25 | n.s. |
| T1 | F - A | +0.013 | +3.11 | +1.39 | ↑ |
| T1 | C - F | +0.030 | +0.40 | +0.18 | n.s. |
| T1 | D - B | +0.055 | +0.72 | +0.32 | n.s. |
| T1 | E - B | +0.019 | +0.22 | +0.10 | n.s. |
| T1 | D - E | +0.036 | +0.99 | +0.44 | n.s. |
| T2 | C - A | +0.085 | +0.68 | +0.30 | n.s. |
| T2 | F - A | -0.015 | -2.32 | -1.04 | n.s. |
| T2 | C - F | +0.100 | +0.80 | +0.36 | n.s. |
| T2 | D - B | +0.335 | +3.08 | +1.38 | ↑ |
| T3 | C - A | +0.011 | +0.15 | +0.07 | n.s. |
| T3 | F - A | +0.009 | +2.02 | +0.91 | n.s. |
| T3 | C - F | +0.002 | +0.03 | +0.01 | n.s. |

## (2) Early v_phys_ratio mean over 5M–50M (PSC groups only)

If PSC's structural prior anchors the critic during cold-start, this should be near 1.0 early.

| Group | T1 | T2 | T3 |
|:---:|:---:|:---:|:---:|
| C | 0.680 ± 0.006 | 1.011 ± 0.022 | 0.681 ± 0.005 |
| D | 0.664 ± 0.009 | 0.983 ± 0.009 | — |
| E | 0.653 ± 0.009 | — | — |

## (3) Time-to-SR threshold — env-steps to first reach SR

Reward-type-independent (SR is computed identically regardless of reward shape).

| Task | Group | SR=0.1 (M steps) | SR=0.2 (M steps) | SR=0.3 (M steps) | SR=0.4 (M steps) | SR=0.5 (M steps) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| T1 | A | 5 | 5 | 9 | 279 | 502 |
| T1 | B | 218 | 752 | 937 (1/5) | never | never |
| T1 | C | 5 | 5 | 9 | 223 | 414 |
| T1 | D | 212 | 601 | 836 (4/5) | 996 (1/5) | never |
| T1 | E | 172 | 563 | 874 | 990 (1/5) | never |
| T1 | F | 5 | 5 | 9 | 253 | 461 |
| T2 | A | 54 | 141 | 821 (3/5) | never | never |
| T2 | B | never | never | never | never | never |
| T2 | C | 42 | 103 | 703 (4/5) | never | never |
| T2 | D | never | never | never | never | never |
| T2 | F | 52 | 105 | 628 (3/5) | never | never |
| T3 | A | 5 | 5 | 9 | 281 | 499 |
| T3 | C | 5 | 5 | 9 | 215 | 412 |
| T3 | F | 5 | 5 | 9 | 243 | 438 |

### Within-reward speedup ratio at SR=0.2 (cold-start exit)

Ratio = baseline / variant. >1.0 means PSC reaches SR=0.2 in fewer steps.

| Task | Contrast | Median steps (a) | Median steps (b) | Speedup b/a |
|:---:|:---:|:---:|:---:|:---:|
| T1 | C vs A | 5M | 5M | 1.00× |
| T1 | F vs A | 5M | 5M | 1.00× |
| T1 | D vs B | 601M | 752M | 1.25× |
| T1 | E vs B | 563M | 752M | 1.34× |
| T2 | C vs A | 103M | 141M | 1.37× |
| T2 | F vs A | 105M | 141M | 1.35× |
| T3 | C vs A | 5M | 5M | 1.00× |
| T3 | F vs A | 5M | 5M | 1.00× |