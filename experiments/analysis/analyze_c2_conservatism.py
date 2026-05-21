"""
C2 — Policy-Level Conservatism Mechanism Analysis (post-A1/A2)

After A1+A2 established that PSC's value-side benefit is cold-start anchoring
(not variance reduction), this script tests the *policy-side* mechanism behind
the +58% OOD safety advantage of D over A (§V-C.2).

Hypothesis: sparse-trained policies (B/D/E) emit smaller control commands
than dense-trained policies (A/C/F) under extreme OOD physics, and this is
what makes them safer.

Metrics computed per (group, seed, lambda) cell:
  - |action|_2 mean / p95
  - per-dim action statistics (roll/pitch/yaw rate, thrust)
  - saturation triggering rate (P(sat > 0.5))
  - per-env action std (intra-episode variability)
  - v_err mean / p95

Paired statistical contrasts at lambda=1.0 (the OOD-safety-critical regime):
  - D vs A, D vs C, D vs F (cross-reward)
  - B vs A (within-MLP cross-reward)
  - C vs A, F vs A (within-dense)
  - D vs B (within-sparse, PSC vs MLP)

Outputs (experiments/_c2_conservatism/):
  c2_table.md
  c2_action_magnitude_curves.{pdf,png}   — |action|_2 mean vs lambda, all 6 groups
  c2_action_p95_curves.{pdf,png}         — p95 magnitude
  c2_saturation_curves.{pdf,png}         — saturation triggering rate
  c2_per_env_std_curves.{pdf,png}        — intra-episode variability
  c2_paired_contrasts.md                 — full paired t-tables at lambda=1.0
"""
import glob
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

# IEEE styling
COL_WIDTH = 3.5
DBL_WIDTH = 7.16
ROW_H = 2.6
rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Liberation Serif"],
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
    "lines.linewidth": 1.5,
    "axes.linewidth": 0.6,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,
})
COLORS = {"A": "#666666", "B": "#cc4444", "C": "#3377cc", "D": "#22aa44",
          "E": "#aa44aa", "F": "#dd8800"}
LABELS = {"A": "A: MLP+dense", "B": "B: MLP+sparse", "C": "C: PSC+dense",
          "D": "D: PSC+sparse", "E": "E: PSC$_{\\mathrm{fix-}w}$+sparse",
          "F": "F: Cai 2025 dual"}

GROUPS = ["A", "B", "C", "D", "E", "F"]
SEEDS = [42, 123, 456, 789, 1024]
LAMBDAS = [0.0, 0.3, 0.5, 0.7, 1.0]

TRAJ_DIR = "experiments/_c2d1/T1_trajectories"
OUT_DIR = "experiments/_c2_conservatism"


def load_cell(g: str, seed: int, lam: float) -> dict:
    path = f"{TRAJ_DIR}/traj_{g}_{seed}_lam{lam:.1f}.npz"
    if not os.path.exists(path):
        return None
    z = np.load(path)
    return {
        "action":     z["action"],
        "saturation": z["saturation"],
        "tilt":       z["tilt"],
        "v_err":      z["v_err"],
        "active":     z["active"],
    }


def per_env_action_std(action: np.ndarray, active: np.ndarray) -> np.ndarray:
    """Per-env L2-norm of std-vector over time (std per action dim, then norm)."""
    T, E, K = action.shape
    out = np.full(E, np.nan)
    for e in range(E):
        m = active[:, e] > 0.5
        if m.sum() >= 2:
            out[e] = np.linalg.norm(action[m, e].std(axis=0))
    return out[~np.isnan(out)]


def cell_metrics(c: dict) -> dict:
    """Compute the per-cell summary metrics. Active-step weighted."""
    if c is None:
        return None
    action = c["action"]
    active = c["active"]
    sat = c["saturation"]
    v_err = c["v_err"]

    mask = active > 0.5
    action_flat = action.reshape(-1, action.shape[-1])[mask.reshape(-1)]
    sat_flat = sat.reshape(-1)[mask.reshape(-1)]
    v_flat = v_err.reshape(-1)[mask.reshape(-1)]

    anorm = np.linalg.norm(action_flat, axis=-1)
    env_std = per_env_action_std(action, active)

    return {
        "anorm_mean":     float(anorm.mean()),
        "anorm_p95":      float(np.percentile(anorm, 95)),
        "anorm_p50":      float(np.percentile(anorm, 50)),
        "thrust_mean":    float(action_flat[:, 3].mean()),
        "thrust_p05":     float(np.percentile(action_flat[:, 3], 5)),
        "roll_rate_p95_abs": float(np.percentile(np.abs(action_flat[:, 0]), 95)),
        "pitch_rate_p95_abs": float(np.percentile(np.abs(action_flat[:, 1]), 95)),
        "yaw_rate_p95_abs": float(np.percentile(np.abs(action_flat[:, 2]), 95)),
        "sat_mean":       float(sat_flat.mean()),
        "sat_trig_rate":  float((sat_flat > 0.5).mean()),
        "env_std_mean":   float(env_std.mean()) if len(env_std) > 0 else float("nan"),
        "env_std_sd":     float(env_std.std()) if len(env_std) > 0 else float("nan"),
        "v_err_mean":     float(v_flat.mean()),
        "v_err_p95":      float(np.percentile(v_flat, 95)),
    }


def paired_t(diffs: np.ndarray) -> Tuple[float, float, float]:
    n = len(diffs)
    if n < 2: return float("nan"), float("nan"), float("nan")
    m = float(np.mean(diffs))
    sd = float(np.std(diffs, ddof=1))
    if sd == 0: return m, float("inf"), float("inf")
    t = m / (sd / np.sqrt(n))
    d = m / sd
    return m, t, d


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load all cells; structure: data[g][seed][lam] = metrics dict
    print("[*] loading trajectory cells...")
    data: Dict[str, Dict[int, Dict[float, dict]]] = {}
    n_loaded = 0
    n_missing = 0
    for g in GROUPS:
        for s in SEEDS:
            for lam in LAMBDAS:
                c = load_cell(g, s, lam)
                if c is None:
                    n_missing += 1
                    continue
                m = cell_metrics(c)
                data.setdefault(g, {}).setdefault(s, {})[lam] = m
                n_loaded += 1
    print(f"[+] loaded {n_loaded} cells; {n_missing} missing")

    METRICS_OF_INTEREST = [
        "anorm_mean", "anorm_p95", "env_std_mean",
        "thrust_mean", "sat_trig_rate", "v_err_mean",
        "roll_rate_p95_abs", "pitch_rate_p95_abs",
    ]

    # ---- Table 1: per-group plateau (lambda=1.0) ----
    lines = []
    lines.append("# C2 — Policy-Level Conservatism Mechanism (post-A1/A2)\n")
    lines.append("Trajectory data from 5 seeds × 5 λ × 100 episodes per group on T1 OOD eval.\n")
    lines.append("Active-step-weighted distributions; all values are 5-seed mean ± SD.\n")

    lines.append("\n## Table C2-I. Per-metric plateau at λ=1.0 (5-seed mean ± SD)\n")
    lines.append("| Group | |action|₂ mean | |action|₂ p95 | env-std mean | thrust mean | sat trig | v_err mean |")
    lines.append("|:-:|:-:|:-:|:-:|:-:|:-:|:-:|")
    for g in GROUPS:
        if g not in data: lines.append(f"| {g} | — | — | — | — | — | — |"); continue
        cells = []
        for metric in ["anorm_mean", "anorm_p95", "env_std_mean", "thrust_mean", "sat_trig_rate", "v_err_mean"]:
            vals = []
            for s in SEEDS:
                if s in data[g] and 1.0 in data[g][s]:
                    vals.append(data[g][s][1.0][metric])
            if not vals:
                cells.append("—")
            else:
                a = np.array(vals)
                cells.append(f"{a.mean():.3f} ± {a.std(ddof=1):.3f}")
        lines.append(f"| {g} | {cells[0]} | {cells[1]} | {cells[2]} | {cells[3]} | {cells[4]} | {cells[5]} |")

    # ---- Within-reward & cross-reward paired contrasts at lambda=1.0 ----
    CONTRASTS = [
        ("D - A", "D", "A", "cross-reward (main C1 anchor)"),
        ("D - C", "D", "C", "cross-reward"),
        ("D - F", "D", "F", "cross-reward (vs Cai)"),
        ("B - A", "B", "A", "cross-reward (within-MLP)"),
        ("E - A", "E", "A", "cross-reward"),
        ("C - A", "C", "A", "within-dense"),
        ("F - A", "F", "A", "within-dense"),
        ("C - F", "C", "F", "within-dense"),
        ("D - B", "D", "B", "within-sparse"),
        ("E - B", "E", "B", "within-sparse"),
        ("D - E", "D", "E", "within-sparse"),
    ]
    lines.append("\n## Table C2-II. Paired contrasts on |action|₂ mean @ λ=1.0\n")
    lines.append("Positive Δ ⇒ variant has higher action magnitude.\n")
    lines.append("| Contrast | Type | Δ |action|₂ | t (df=4) | Cohen d | p<0.05? |")
    lines.append("|:--:|:--:|:--:|:--:|:--:|:--:|")
    for label, ga, gb, kind in CONTRASTS:
        diffs = []
        for s in SEEDS:
            if (s in data.get(ga, {}) and 1.0 in data[ga][s]
                    and s in data.get(gb, {}) and 1.0 in data[gb][s]):
                diffs.append(data[ga][s][1.0]["anorm_mean"] - data[gb][s][1.0]["anorm_mean"])
        if len(diffs) < 2: continue
        m, t, d = paired_t(np.array(diffs))
        sig = "✓" if abs(t) > 2.776 else "n.s."
        lines.append(f"| {label} | {kind} | {m:+.4f} | {t:+.2f} | {d:+.2f} | {sig} |")

    # Repeat for env_std_mean and sat_trig_rate
    for metric, mname in [("env_std_mean", "per-env action std"), ("sat_trig_rate", "saturation triggering rate")]:
        lines.append(f"\n## Table C2-III/IV. Paired contrasts on {mname} @ λ=1.0\n")
        lines.append(f"| Contrast | Type | Δ {metric} | t (df=4) | Cohen d | p<0.05? |")
        lines.append("|:--:|:--:|:--:|:--:|:--:|:--:|")
        for label, ga, gb, kind in CONTRASTS:
            diffs = []
            for s in SEEDS:
                if (s in data.get(ga, {}) and 1.0 in data[ga][s]
                        and s in data.get(gb, {}) and 1.0 in data[gb][s]):
                    diffs.append(data[ga][s][1.0][metric] - data[gb][s][1.0][metric])
            if len(diffs) < 2: continue
            m, t, d = paired_t(np.array(diffs))
            sig = "✓" if abs(t) > 2.776 else "n.s."
            lines.append(f"| {label} | {kind} | {m:+.4f} | {t:+.2f} | {d:+.2f} | {sig} |")

    # ---- Per-lambda trends table ----
    lines.append("\n## Table C2-V. |action|₂ mean vs λ (5-seed mean)\n")
    lines.append("| Group | λ=0.0 | λ=0.3 | λ=0.5 | λ=0.7 | λ=1.0 |")
    lines.append("|:-:|:-:|:-:|:-:|:-:|:-:|")
    for g in GROUPS:
        if g not in data: continue
        cells = []
        for lam in LAMBDAS:
            vals = [data[g][s][lam]["anorm_mean"] for s in SEEDS
                    if s in data[g] and lam in data[g][s]]
            if not vals: cells.append("—")
            else: cells.append(f"{np.mean(vals):.3f}")
        lines.append(f"| {g} | " + " | ".join(cells) + " |")

    with open(f"{OUT_DIR}/c2_table.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[+] wrote {OUT_DIR}/c2_table.md")

    # ---- Figures ----
    def plot_metric_vs_lambda(metric: str, ylabel: str, fname: str):
        fig, ax = plt.subplots(1, 1, figsize=(COL_WIDTH, ROW_H))
        for g in GROUPS:
            if g not in data: continue
            mvals = []
            svals = []
            for lam in LAMBDAS:
                vals = [data[g][s][lam][metric] for s in SEEDS
                        if s in data[g] and lam in data[g][s]]
                if not vals:
                    mvals.append(np.nan)
                    svals.append(np.nan)
                else:
                    mvals.append(float(np.mean(vals)))
                    svals.append(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0)
            mvals = np.array(mvals); svals = np.array(svals)
            ax.errorbar(LAMBDAS, mvals, yerr=svals, label=LABELS[g],
                        color=COLORS[g], lw=1.3, capsize=2, marker='o', markersize=4)
        ax.set_xlabel(r"Test-time DR strength $\lambda$")
        ax.set_ylabel(ylabel)
        ax.legend(loc="best", framealpha=0.85, edgecolor='none', fontsize=7)
        ax.set_xlim(-0.05, 1.05)
        plt.tight_layout(pad=0.5)
        fig.savefig(f"{OUT_DIR}/{fname}.pdf"); fig.savefig(f"{OUT_DIR}/{fname}.png", dpi=160)
        plt.close(fig)
        print(f"[+] wrote {OUT_DIR}/{fname}.pdf")

    plot_metric_vs_lambda("anorm_mean", r"$|action|_2$ mean", "c2_action_magnitude_curves")
    plot_metric_vs_lambda("anorm_p95", r"$|action|_2$ p95", "c2_action_p95_curves")
    plot_metric_vs_lambda("env_std_mean", "Per-env action std", "c2_per_env_std_curves")
    plot_metric_vs_lambda("sat_trig_rate", r"$P(\mathrm{sat} > 0.5)$", "c2_saturation_curves")
    plot_metric_vs_lambda("thrust_mean", "Thrust mean", "c2_thrust_curves")
    plot_metric_vs_lambda("v_err_mean", r"$|v_{\mathrm{err}}|$ mean (m/s)", "c2_v_err_curves")

    # ---- Verdict ----
    print()
    print("=" * 78)
    print("C2 verdict — policy-level conservatism mechanism at λ=1.0")
    print("=" * 78)
    print()
    print("Key cross-reward contrasts (D vs each dense baseline):")
    for label, ga, gb, kind in [("D - A", "D", "A", ""), ("D - C", "D", "C", ""), ("D - F", "D", "F", "")]:
        for metric in ["anorm_mean", "env_std_mean", "sat_trig_rate"]:
            diffs = []
            for s in SEEDS:
                if (s in data.get(ga, {}) and 1.0 in data[ga][s]
                        and s in data.get(gb, {}) and 1.0 in data[gb][s]):
                    diffs.append(data[ga][s][1.0][metric] - data[gb][s][1.0][metric])
            if len(diffs) < 2: continue
            m, t, d = paired_t(np.array(diffs))
            base = np.mean([data[gb][s][1.0][metric] for s in SEEDS if s in data.get(gb, {}) and 1.0 in data[gb][s]])
            relpct = (m / base * 100.0) if base != 0 else float("nan")
            sig = "***" if abs(t) > 4.604 else ("**" if abs(t) > 2.776 else ("*" if abs(t) > 2.132 else "n.s."))
            print(f"  {label}  {metric:18s}  Δ={m:+.4f}  ({relpct:+.1f}%)  t={t:+.2f}  d={d:+.2f}  {sig}")
        print()


if __name__ == "__main__":
    main()
