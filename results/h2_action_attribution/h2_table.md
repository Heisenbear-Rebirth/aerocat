# H2 — Action spectrum & state-action lag attribution

Assuming control rate 50 Hz; bands: low=(0.5, 5.0), mid=(5.0, 15.0), high=(15.0, 25.0) Hz.

Per-channel: roll_rate, pitch_rate, yaw_rate, thrust.


## (1) High-band fraction E_high/E_total per channel at λ=1.0 (mean ± SD over 5 seeds)

Higher value = more high-frequency content in the command. Lower = smoother control.

| Group | roll | pitch | yaw | thrust |
|:---:|:---:|:---:|:---:|:---:|
| A | 0.001±0.000 | 0.001±0.000 | 0.002±0.000 | 0.003±0.001 |
| B | 0.001±0.000 | 0.001±0.000 | 0.002±0.001 | 0.006±0.003 |
| C | 0.001±0.000 | 0.001±0.000 | 0.002±0.000 | 0.004±0.001 |
| D | 0.001±0.000 | 0.001±0.000 | 0.003±0.001 | 0.006±0.002 |
| E | 0.001±0.000 | 0.002±0.000 | 0.003±0.001 | 0.005±0.001 |
| F | 0.001±0.000 | 0.001±0.000 | 0.002±0.000 | 0.004±0.000 |

## (2) Paired Δ(high-band fraction) per channel at λ=1.0

Negative ⇒ sparse policy has SMOOTHER (lower high-freq) commands.

| Contrast | roll | pitch | yaw | thrust |
|:---:|:---:|:---:|:---:|:---:|
| D − A | +0.000 (t=+1.7) | +0.000** (t=+3.6) | +0.001** (t=+5.9) | +0.002 (t=+2.0) |
| D − B | -0.000 (t=-0.9) | +0.000 (t=+0.1) | +0.001 (t=+1.7) | -0.001 (t=-0.7) |
| D − E | -0.000 (t=-1.1) | -0.000 (t=-0.3) | +0.000 (t=+1.2) | +0.001 (t=+0.9) |
| B − A | +0.000· (t=+2.5) | +0.000** (t=+7.4) | +0.001 (t=+1.4) | +0.003· (t=+2.7) |
| C − A | -0.000 (t=-1.4) | -0.000 (t=-1.8) | -0.000· (t=-2.2) | +0.001 (t=+1.7) |
| F − A | +0.000 (t=+0.1) | -0.000· (t=-2.8) | -0.000 (t=-0.7) | +0.000 (t=+0.9) |

** = p<0.05 (|t|>2.776, df=4); · = p<0.10 (|t|>2.132).


## (3) Low-band fraction E_low/E_total per channel at λ=1.0

Higher = more steady-state/trim content; lower = more transient/corrective.

| Group | roll | pitch | yaw | thrust |
|:---:|:---:|:---:|:---:|:---:|
| A | 0.283±0.017 | 0.274±0.014 | 0.289±0.011 | 0.232±0.016 |
| B | 0.255±0.051 | 0.246±0.052 | 0.276±0.064 | 0.267±0.075 |
| C | 0.285±0.019 | 0.274±0.009 | 0.274±0.011 | 0.250±0.022 |
| D | 0.306±0.065 | 0.255±0.069 | 0.273±0.071 | 0.277±0.061 |
| E | 0.261±0.026 | 0.237±0.025 | 0.233±0.028 | 0.249±0.027 |
| F | 0.284±0.014 | 0.292±0.012 | 0.274±0.018 | 0.234±0.016 |

## (4) State-action peak-lag-correlation: median lag (steps) and corr at λ=1.0

Lagged Pearson(|action|_2[t-k], v_err[t]); peak |corr| over k∈[0,20].

Higher lag ⇒ action responds to v_err with delay (anticipatory? or sluggish?).

| Group | median peak lag (steps) | median |peak corr| | n |
|:---:|:---:|:---:|:---:|
| A | 9.2±1.5 | -0.103±0.142 | 5 |
| B | 8.4±0.7 | -0.067±0.149 | 5 |
| C | 9.6±2.0 | -0.171±0.041 | 5 |
| D | 7.8±1.5 | +0.088±0.169 | 5 |
| E | 7.9±2.0 | +0.090±0.168 | 5 |
| F | 7.9±1.5 | -0.171±0.030 | 5 |