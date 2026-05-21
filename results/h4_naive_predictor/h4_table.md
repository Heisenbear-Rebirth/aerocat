# H4 — Naïve physical-signal AUROC for crash prediction (K=20 steps)

Label: 'tilt > 1.5 within K=20 steps'. Predictors: bare tilt, saturation, v_err, tilt+0.1·v_err.

**Compare to H9's V_phys AUROC** (0.80-0.83 in PSC groups at λ=1.0).

Random = 0.5. If naïve tilt achieves > 0.95, H9's PSC-interpretability claim is weakened.


## Predictor: tilt

AUROC mean ± SD across 5 seeds, per (group, λ); direction in parens.

| Group | λ=0.0 | λ=0.3 | λ=0.5 | λ=0.7 | λ=1.0 |
|:---:|:---:|:---:|:---:|:---:|:---:|
| A | 0.953±0.009 (high) | 0.853±0.013 (high) | 0.838±0.017 (high) | 0.805±0.011 (high) | 0.754±0.021 (high) |
| B | 0.930±0.071 (high) | 0.886±0.029 (high) | 0.882±0.018 (high) | 0.828±0.012 (high) | 0.770±0.008 (high) |
| C | 0.957±0.003 (high) | 0.859±0.010 (high) | 0.822±0.012 (high) | 0.793±0.023 (high) | 0.751±0.017 (high) |
| D | 0.950±0.036 (high) | 0.869±0.014 (high) | 0.855±0.023 (high) | 0.811±0.023 (high) | 0.750±0.009 (high) |
| E | 0.944±0.046 (high) | 0.867±0.030 (high) | 0.850±0.029 (high) | 0.808±0.008 (high) | 0.737±0.009 (high) |
| F | 0.963±0.011 (high) | 0.866±0.019 (high) | 0.836±0.016 (high) | 0.810±0.010 (high) | 0.753±0.007 (high) |

## Predictor: saturation

AUROC mean ± SD across 5 seeds, per (group, λ); direction in parens.

| Group | λ=0.0 | λ=0.3 | λ=0.5 | λ=0.7 | λ=1.0 |
|:---:|:---:|:---:|:---:|:---:|:---:|
| A | 0.972±0.003 (high) | 0.933±0.012 (high) | 0.832±0.025 (high) | 0.743±0.034 (high) | 0.637±0.019 (high) |
| B | 0.928±0.046 (high) | 0.864±0.040 (high) | 0.761±0.026 (high) | 0.721±0.027 (high) | 0.656±0.015 (high) |
| C | 0.970±0.005 (high) | 0.940±0.004 (high) | 0.842±0.017 (high) | 0.746±0.017 (high) | 0.639±0.014 (high) |
| D | 0.851±0.120 (high) | 0.808±0.051 (high) | 0.722±0.051 (high) | 0.671±0.032 (high) | 0.628±0.028 (high) |
| E | 0.893±0.087 (high) | 0.872±0.030 (high) | 0.759±0.024 (high) | 0.713±0.012 (high) | 0.654±0.023 (high) |
| F | 0.973±0.004 (high) | 0.928±0.010 (high) | 0.849±0.024 (high) | 0.761±0.023 (high) | 0.670±0.015 (high) |

## Predictor: v_err

AUROC mean ± SD across 5 seeds, per (group, λ); direction in parens.

| Group | λ=0.0 | λ=0.3 | λ=0.5 | λ=0.7 | λ=1.0 |
|:---:|:---:|:---:|:---:|:---:|:---:|
| A | 0.957±0.010 (high) | 0.940±0.021 (high) | 0.880±0.015 (high) | 0.798±0.020 (high) | 0.670±0.009 (high) |
| B | 0.639±0.133 (high) | 0.794±0.050 (high) | 0.701±0.037 (high) | 0.651±0.019 (high) | 0.557±0.027 (high) |
| C | 0.958±0.005 (high) | 0.941±0.005 (high) | 0.883±0.009 (high) | 0.794±0.009 (high) | 0.664±0.009 (high) |
| D | 0.890±0.046 (high) | 0.784±0.066 (high) | 0.709±0.053 (high) | 0.647±0.018 (high) | 0.572±0.021 (high) |
| E | 0.921±0.019 (high) | 0.816±0.037 (high) | 0.724±0.018 (high) | 0.653±0.020 (high) | 0.578±0.004 (high) |
| F | 0.960±0.007 (high) | 0.957±0.008 (high) | 0.896±0.012 (high) | 0.807±0.008 (high) | 0.685±0.007 (high) |

## Predictor: tilt_plus_verr

AUROC mean ± SD across 5 seeds, per (group, λ); direction in parens.

| Group | λ=0.0 | λ=0.3 | λ=0.5 | λ=0.7 | λ=1.0 |
|:---:|:---:|:---:|:---:|:---:|:---:|
| A | 0.977±0.004 (high) | 0.953±0.008 (high) | 0.921±0.010 (high) | 0.872±0.007 (high) | 0.783±0.012 (high) |
| B | 0.859±0.151 (high) | 0.949±0.011 (high) | 0.916±0.013 (high) | 0.847±0.010 (high) | 0.748±0.016 (high) |
| C | 0.978±0.002 (high) | 0.951±0.007 (high) | 0.912±0.006 (high) | 0.866±0.016 (high) | 0.780±0.009 (high) |
| D | 0.962±0.017 (high) | 0.921±0.012 (high) | 0.883±0.003 (high) | 0.827±0.013 (high) | 0.734±0.010 (high) |
| E | 0.970±0.015 (high) | 0.916±0.015 (high) | 0.886±0.013 (high) | 0.825±0.005 (high) | 0.728±0.008 (high) |
| F | 0.981±0.003 (high) | 0.964±0.007 (high) | 0.929±0.013 (high) | 0.882±0.007 (high) | 0.788±0.006 (high) |

## Head-to-head AUROC at λ=1.0 (PSC groups only, K=20)

Compare each naïve predictor's mean AUROC to H9's reported V_phys AUROC.

| Group | tilt (H4) | saturation (H4) | v_err (H4) | tilt+0.1·v_err (H4) | **V_phys (H9 crash)** | **V_phys (H9 imm)** |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| C | 0.751 | 0.639 | 0.664 | 0.780 | **0.810** | **0.668** |
| D | 0.750 | 0.628 | 0.572 | 0.734 | **0.833** | **0.754** |
| E | 0.737 | 0.654 | 0.578 | 0.728 | **0.796** | **0.738** |

*H4 label = tilt-exceedance within K; H9 label = done-with-reward<0 (crash) or any-done (imm). Different label definitions, similar OOD severity.*
