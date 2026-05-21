# AeroCat

> Reproduction package for an empirical study of out-of-distribution safety in PPO-LSTM quadrotor reinforcement learning.

This repository contains the JAX/Flax implementation of a multi-task quadrotor
reinforcement-learning environment, training pipeline (PPO + recurrent
actor-critic with LSTM and an optional physics-structured critic, PSC), and a
complete suite of evaluation and analysis scripts used to produce the 15
diagnostic experiments documented in [`docs/COMPLETE_DELIVERABLE.md`](docs/COMPLETE_DELIVERABLE.md).

The accompanying paper investigates **which factors actually drive
out-of-distribution (OOD) safety in quadrotor RL** under randomised mass,
inertia, wind, turbulence, sensor bias, actuator efficiency loss, initial-state
perturbations, and impulsive collisions. The empirical headline:

- **OOD crashes are collision-singular.** Of seven OOD perturbation axes
  (mass / wind / turbulence / sensor / actuator / initial-state / collision),
  only impulsive collisions produce flipping failures (≥ 0.1% per-episode
  crash) across all six policy variants tested. The other six axes degrade
  tracking error but cause zero crashes when activated in isolation.
- **Sparse-reward policies appear ~2× safer per episode but ~10× *less* safe per
  flight-second.** The per-mission advantage of sparse-reward training operates
  via shorter exposure (rapid goal-directed termination), not per-step
  collision robustness. We report both metrics openly.
- **The PSC critic acts as a cold-start anchor, not a "better critic".** PSC
  raises steady-state TD-error standard deviation by 7–25 % across all three
  tasks. It accelerates sparse-reward `SR=0.2` exit by 25–37 % via a non-flat
  baseline that keeps `advantage ≠ 0` while reward is still ≈ 0. This is an
  explicit bias–variance trade-off, not a Pareto improvement.

These findings, and the negative results that triangulate them, are reproduced
end-to-end by the scripts in this repository.

---

## Repository layout

```
aerocat/
├── README.md                      ← this file
├── LICENSE                        ← MIT
├── pyproject.toml                 ← installable as `aerocat`
├── requirements.txt               ← pinned core dependencies
├── src/aerocat/                   ← main Python package
│   ├── config.py                  ← TrainConfig, AblationConfig, CurriculumConfig
│   ├── core/                      ← state, replay, low-level primitives
│   ├── envs/                      ← quadrotor env (uav_env.py) + reward modes
│   ├── physics/                   ← rigid-body dynamics, turbulence, motors
│   ├── generators/                ← domain-randomised parameter sampling
│   ├── networks/                  ← StochasticActorCritic + PSC critic head
│   ├── training/                  ← PPO trainer (ppo_trainer.py)
│   ├── tasks/                     ← T1 velocity / T2 waypoint / T3 disturbance
│   ├── control/                   ← cascade L1/L3 controllers (baseline outer loop)
│   ├── utils/                     ← checkpoint manager, logging helpers
│   └── scripts/                   ← train + eval CLI entry points
│       ├── run_ablation.py        ← train one (group, seed, task) for 1 B steps
│       ├── run_dr_eval.py         ← OOD eval (deterministic policy)
│       ├── run_dr_eval_stoch.py   ← OOD eval (stochastic policy)
│       ├── run_dr_eval_traj.py    ← OOD eval + action/saturation/tilt dump
│       ├── run_dr_eval_calib.py   ← OOD eval + V/V_phys/V_res dump
│       ├── run_dr_eval_perdim.py  ← per-OOD-dimension decomposition (with --resume)
│       └── train.py / test_*.py   ← smoke tests
├── experiments/
│   ├── analysis/                  ← 20 analyze_*.py scripts (one per diagnostic)
│   └── launchers/                 ← bash launchers for long sweeps
└── results/                       ← committed markdown tables + figures
                                     from the 14 reported diagnostics
└── docs/
    └── COMPLETE_DELIVERABLE.md    ← canonical single-source experimental log
                                     (numbers, paths, contrasts, t-stats)
```

`src/aerocat/` is a clean Python package; nothing in `experiments/` is imported
from it. The analysis scripts are standalone — they read JSON / npz / metrics
artefacts produced by the training and eval CLIs and write markdown tables and
matplotlib figures into `experiments_runtime/<diagnostic_id>/`.

---

## Quick start

### 1. Hardware and OS

- **GPU**: NVIDIA RTX 50-series (Blackwell, sm\_120) was used for the reported
  runs. JAX ≥ 0.6.0 wheels include sm\_120; earlier JAX wheels will fail to
  load on Blackwell.
- **OS**: Tested on Windows 11 + WSL 2 (Ubuntu 22.04). Pure Linux should work
  identically; macOS is untested.

### 2. Install

```bash
# clone
git clone https://github.com/Heisenbear-Rebirth/aerocat.git
cd aerocat

# create env (Python ≥ 3.10)
python -m venv .venv
source .venv/bin/activate          # Linux / WSL
# or  .venv\Scripts\activate.bat    on Windows

# install
pip install --upgrade pip
pip install -e .
```

The `pip install -e .` step makes `aerocat` importable from anywhere in the
env, so `python -m aerocat.scripts.run_ablation ...` works from the repo root.

### 3. Smoke test (≤ 1 min on a single GPU)

```bash
python -m aerocat.scripts.test_training
```

This runs a couple of PPO iterations on the velocity task and prints the SR /
return numbers — no checkpoint is saved.

---

## Reproducing the experimental suite

Throughout, set `AEROCAT_EXP_BASE=$PWD/experiments_runtime` so every script
reads and writes under that one directory. All paths below are relative to the
repo root.

### A. Train the six 1 B-step policies

| ID | Group | Critic | Reward | Notes |
|:---:|:-----:|:-------|:-------|:------|
| A | MLP-dense        | MLP                            | dense  | baseline |
| B | MLP-sparse       | MLP                            | sparse | reward ablation |
| C | PSC-dense        | PSC (5 learnable scalars)      | dense  | |
| D | PSC-sparse       | PSC (5 learnable scalars)      | sparse | **main proposed config** |
| E | PSCfixedw-sparse | PSC with frozen scalars        | sparse | mechanism ablation |
| F | Cai 2025 dual    | dual-critic baseline           | dense  | SOTA comparison |

For each `(group, seed) ∈ {A,B,C,D,E,F} × {42, 123, 456, 789, 1024}`:

```bash
python -m aerocat.scripts.run_ablation \
    --group D --task velocity --seed 42 \
    --output-dir experiments_runtime/ablation_D_sparse_psc/seed_42
```

Wall-clock: ~1.18 h per run on RTX 5090 (~235 k env-steps / s). The full
6 × 5 = 30 runs for T1 takes ~35 GPU-hours. T2 / T3 each add a similar amount
(but only 5 / 4 groups). The E2 leave-one-out study adds 5 × 5 = 25 runs on
top.

### B. OOD evaluation (cheap, requires checkpoints)

```bash
# P0: main OOD curve, deterministic policy, 100 episodes per (group, seed, λ)
python -m aerocat.scripts.run_dr_eval \
    --groups A B C D E F \
    --seeds 42 123 456 789 1024 \
    --lambdas 0.0 0.3 0.5 0.7 1.0 \
    --task velocity \
    --output experiments_runtime/_dr_eval_results.json

# C3: stochastic-policy OOD validation
python -m aerocat.scripts.run_dr_eval_stoch  ...

# C1: deterministic eval + dump V / V_phys / V_res / reward / done
python -m aerocat.scripts.run_dr_eval_calib  ...

# C2/D1: deterministic eval + dump action / saturation / tilt / v_err
python -m aerocat.scripts.run_dr_eval_traj   ...

# H5: per-dimension OOD eval on T1
python -m aerocat.scripts.run_dr_eval_perdim \
    --task velocity \
    --groups A B C D E F \
    --dims nominal mass wind turb sensor actuator init_state collision all_l1

# H8: same on T3 (cross-task validation)
python -m aerocat.scripts.run_dr_eval_perdim \
    --task disturbance --groups A C F D \
    --dims nominal mass wind turb sensor actuator init_state collision all_l1
```

Each OOD eval takes ~30 s per `(group, seed, condition)` cell on RTX 5090. The
six runs above cost a total of ~6 GPU-hours.

`run_dr_eval_perdim.py` supports `--resume`: rerun the same command and it
will skip cells already present in the output JSON. We used this when WSL
killed long sweeps mid-run.

### C. Analyses (CPU only, < 1 min each)

After training and the relevant eval pass, run any of:

```bash
python experiments/analysis/analyze_a1_critic_variance.py
python experiments/analysis/analyze_a2_cold_start.py
python experiments/analysis/analyze_c1_calibration.py
python experiments/analysis/analyze_c2_conservatism.py
python experiments/analysis/analyze_c3_stochastic.py
python experiments/analysis/analyze_e1_transfer.py
python experiments/analysis/analyze_e2_per_basis.py
python experiments/analysis/analyze_b1plus_drift_vs_leaveoneout.py
python experiments/analysis/analyze_h1_vphys_evolution.py
python experiments/analysis/analyze_h2_action_attribution.py
python experiments/analysis/analyze_h4_naive_predictor.py
python experiments/analysis/analyze_h5_dim_decomposition.py
python experiments/analysis/analyze_h5_dim_decomposition.py \
    --input experiments_runtime/_h8_perdim_t3/results.json \
    --output-dir experiments_runtime/_h8_perdim_t3 \
    --groups A C F D --title-tag H8
python experiments/analysis/analyze_h9_vphys_predictor.py
```

Each script writes `<id>_table.md` plus `<id>_*.pdf` / `<id>_*.png` into its
output directory. A committed copy of every result table and figure is kept
in [`results/`](results/) so you can compare your reruns against the published
numbers without spending the ~50 GPU-hours.

### D. Bash launchers

The `experiments/launchers/run_*.sh` scripts wrap the long-running sweeps used
in the paper. They are committed for transparency but use environment-specific
detach patterns (`setsid nohup ...`); adapt to your job scheduler as needed.

---

## Experiments at a glance

| ID | Question | Where the verdict lives |
|:---:|:---------|:------------------------|
| A1 | Does PSC reduce TD-error variance (the v19.3 control-variate claim)? | [`results/a1_variance/a1_table.md`](results/a1_variance/a1_table.md) — **falsified**, PSC ↑ TD-std 7–25 % |
| A2 | Is PSC a cold-start anchor instead? | [`results/a2_cold_start/a2_table.md`](results/a2_cold_start/a2_table.md) — yes; SR=0.2 reached 25–37 % earlier |
| B1+ | Does RL spontaneously prune PSC basis weights? | [`results/b1plus_drift_vs_e2/b1plus_table.md`](results/b1plus_drift_vs_e2/b1plus_table.md) — w₂,w₃,w₄ drift to ~0; w₀ stays |
| C1 | Does PSC's V stay better-calibrated under OOD? | [`results/c1_calibration/c1_table.md`](results/c1_calibration/c1_table.md) — direction-correct, not PSC-unique |
| C2 | Do sparse-reward policies emit smaller commands? | [`results/c2_conservatism/c2_table.md`](results/c2_conservatism/c2_table.md) — yes, d ≈ -1.9 to -5.9, but D≈B≈E |
| C3 | Does the sparse safety gap survive stochastic policy? | [`results/c3_stochastic/c3_table.md`](results/c3_stochastic/c3_table.md) — yes, 7/7 contrasts retain 108–166 % |
| D1 | Are sparse-trained actions smoother (lower jerk RMS)? | jerk reversed, +18 % — refuted (see C2 table) |
| E1 | Does cross-task PSC weight transfer help cold-start? | [`results/e1_transfer/e1_table.md`](results/e1_transfer/e1_table.md) — no; SR=0.10 reached 28.5 M steps later |
| E2 | Is each of the 5 PSC basis individually necessary? | [`results/e2_per_basis/e2_table.md`](results/e2_per_basis/e2_table.md) — 0/5; removing φ_vel even helps |
| H1 | Does the deployment-time V_phys ratio rise under OOD? | [`results/h1_vphys_evolution/h1_table.md`](results/h1_vphys_evolution/h1_table.md) — only for PSC-dense (C: 0.21 → 0.37, t = +95) |
| H2 | Are sparse commands spectrally smoother / lower-bandwidth? | [`results/h2_action_attribution/h2_table.md`](results/h2_action_attribution/h2_table.md) — no, all groups < 1 % above 15 Hz |
| H4 | How does V_phys compare to a naïve tilt-based crash predictor? | [`results/h4_naive_predictor/h4_table.md`](results/h4_naive_predictor/h4_table.md) — V_phys beats best naïve scalar by 3–8 AUROC pp |
| H5 | Which OOD axis actually drives crashes (T1)? | [`results/h5_perdim/h5_table.md`](results/h5_perdim/h5_table.md) — **collision-singular**; per-second hazard reverses sparse/dense |
| H8 | Does that pattern hold on the T3 disturbance task? | [`results/h8_perdim_t3/h8_table.md`](results/h8_perdim_t3/h8_table.md) — yes, near-perfect cross-task replication |
| H9 | Can V_phys serve as a deployment-time crash predictor? | [`results/h9_vphys_predictor/h9_table.md`](results/h9_vphys_predictor/h9_table.md) — AUROC 0.80–0.83 at λ=1; modest |

Full numerical detail (paired t-statistics, Cohen's d, all per-cell values) is
in [`docs/COMPLETE_DELIVERABLE.md`](docs/COMPLETE_DELIVERABLE.md).

---

## Citation

A paper accompanying this code is in preparation. Until it appears, please
cite the repository:

```bibtex
@misc{aerocat2026,
  title  = {AeroCat: an empirical study of out-of-distribution safety
            in PPO-LSTM quadrotor reinforcement learning},
  author = {AeroCat contributors},
  year   = {2026},
  url    = {https://github.com/Heisenbear-Rebirth/aerocat},
  note   = {Reproduction package}
}
```

---

## License

MIT. See [`LICENSE`](LICENSE).
