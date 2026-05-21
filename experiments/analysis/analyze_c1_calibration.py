"""
C1 — OOD V calibration analysis.

For each (group, seed, lambda) cell, reconstruct the observed return G_t per
(step, env) from the dumped trajectory, then quantify how well V_theta(s_t)
predicts G_t. Hypothesis: PSC's analytic V_phys keeps V calibrated under OOD,
because V_phys does not depend on training-distribution interpolation.

Per-cell metrics on (V, G) restricted to valid steps (those followed by a
within-buffer episode termination, so G is fully observed):
  - Pearson corr(V, G)             — does V predict G's ordering?
  - bias = mean(V - G)              — systematic over/under-estimation
  - RMSE                            — scale of error
  - explained_variance = 1 - Var(G-V)/Var(G)

Also compute the same on (V_phys, G) and (V_res, G) for PSC groups, to
quantify each component's calibration contribution.

Key contrasts at lambda=1.0 (OOD-extreme):
  PSC vs MLP within-reward: C-A (dense), D-B (sparse), E-B (sparse fix-w)

Output:
  experiments/_c1_calibration/
    c1_table.md
    c1_calibration_curves.{pdf,png}  — metrics vs lambda, per group
"""
import glob
import os
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

COL_WIDTH = 3.5
DBL_WIDTH = 7.16
ROW_H = 2.6
rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Liberation Serif"],
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 10,
    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.grid": True, "grid.alpha": 0.3, "grid.linewidth": 0.5,
    "lines.linewidth": 1.5, "axes.linewidth": 0.6,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "pdf.fonttype": 42,
})
COLORS = {"A": "#666666", "B": "#cc4444", "C": "#3377cc", "D": "#22aa44",
          "E": "#aa44aa", "F": "#dd8800"}
LABELS = {"A": "A:MLP+dense", "B": "B:MLP+sparse", "C": "C:PSC+dense",
          "D": "D:PSC+sparse", "E": "E:PSCfix+sparse", "F": "F:Cai dual"}

GROUPS = ["A", "B", "C", "D", "E", "F"]
SEEDS = [42, 123, 456, 789, 1024]
LAMBDAS = [0.0, 0.3, 0.5, 0.7, 1.0]
GAMMA = 0.995

TRAJ_DIR = "experiments/_c1/T1_calib_trajectories"
OUT = "experiments/_c1_calibration"


def reconstruct_G(reward: np.ndarray, done: np.ndarray, gamma: float = GAMMA
                  ) -> Tuple[np.ndarray, np.ndarray]:
    """Per-env backward sweep to reconstruct observed return G_t.
    Valid mask = True for (t, e) where there's a done event at some t' >= t
    within the buffer (so G is fully observed, not truncated)."""
    T, E = reward.shape
    G = np.zeros((T, E), dtype=np.float64)
    valid = np.zeros((T, E), dtype=bool)
    for e in range(E):
        done_idx = np.where(done[:, e] == 1)[0]
        if len(done_idx) == 0:
            continue
        last_done = int(done_idx[-1])
        G[last_done, e] = reward[last_done, e]
        valid[last_done, e] = True
        for t in range(last_done - 1, -1, -1):
            if done[t, e] == 1:
                G[t, e] = reward[t, e]
            else:
                G[t, e] = reward[t, e] + gamma * G[t + 1, e]
            valid[t, e] = True
    return G, valid


def calibration_metrics(V: np.ndarray, G: np.ndarray) -> dict:
    """Pearson corr, bias, RMSE, explained_variance on flat (V, G) pairs."""
    if len(V) < 2:
        return {"corr": np.nan, "bias": np.nan, "rmse": np.nan, "ev": np.nan, "n": len(V)}
    V = V.astype(np.float64)
    G = G.astype(np.float64)
    # Pearson corr
    vm = V.mean(); gm = G.mean()
    vv = V - vm; gg = G - gm
    denom = np.sqrt((vv ** 2).sum() * (gg ** 2).sum())
    corr = float((vv * gg).sum() / denom) if denom > 0 else np.nan
    bias = float((V - G).mean())
    rmse = float(np.sqrt(((V - G) ** 2).mean()))
    var_g = float(G.var())
    ev = float(1.0 - ((G - V).var() / var_g)) if var_g > 0 else np.nan
    return {"corr": corr, "bias": bias, "rmse": rmse, "ev": ev, "n": len(V)}


def process_cell(path: str) -> dict:
    z = np.load(path)
    value = z["value"]
    v_phys = z["v_phys"]
    v_res = z["v_res"]
    reward = z["reward"]
    done = z["done"]
    G, valid = reconstruct_G(reward, done)
    mask = valid.reshape(-1)
    out = {
        "n_steps": value.shape[0],
        "n_envs": value.shape[1],
        "n_valid": int(mask.sum()),
        "V_metrics": calibration_metrics(value.reshape(-1)[mask], G.reshape(-1)[mask]),
        "Vphys_metrics": calibration_metrics(v_phys.reshape(-1)[mask], G.reshape(-1)[mask]),
        "Vres_metrics": calibration_metrics(v_res.reshape(-1)[mask], G.reshape(-1)[mask]),
        "G_mean": float(G.reshape(-1)[mask].mean()) if mask.any() else np.nan,
        "G_std": float(G.reshape(-1)[mask].std()) if mask.any() else np.nan,
        "Vphys_share": float(np.abs(v_phys).mean() / (np.abs(value).mean() + 1e-9)),
    }
    return out


def paired_t(diffs):
    n = len(diffs)
    if n < 2:
        return float("nan"), float("nan"), float("nan")
    m = float(np.mean(diffs))
    sd = float(np.std(diffs, ddof=1))
    if sd == 0:
        return m, float("inf"), float("inf")
    return m, m / (sd / np.sqrt(n)), m / sd


def sig(t):
    at = abs(t)
    return "***" if at > 4.604 else "**" if at > 2.776 else "*" if at > 2.132 else "n.s."


def main():
    os.makedirs(OUT, exist_ok=True)

    print("[*] loading and processing trajectory cells...")
    cells: Dict = {}
    n_done = 0
    for g in GROUPS:
        for s in SEEDS:
            for lam in LAMBDAS:
                path = f"{TRAJ_DIR}/traj_{g}_{s}_lam{lam:.1f}.npz"
                if not os.path.exists(path):
                    continue
                cells[(g, s, lam)] = process_cell(path)
                n_done += 1
    print(f"[+] processed {n_done} cells")

    lines = []
    lines.append("# C1 — OOD V Calibration Analysis (deterministic policy)\n")
    lines.append("Observed return G reconstructed per-env via backward sweep "
                 f"(gamma={GAMMA}), masking trajectory tails that lacked a "
                 "within-buffer episode termination.\n")
    lines.append("Metrics on full V_theta(s) = V_phys + V_res; auxiliary metrics "
                 "on V_phys and V_res alone (PSC groups only).\n")

    # Table I: V calibration metrics at lambda=1.0
    lines.append("\n## Table C1-I. V_theta calibration @ lambda=1.0 (5-seed mean)\n")
    lines.append("| Group | n_valid | corr(V,G) | bias | RMSE | explained_var | G_mean | G_std |")
    lines.append("|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|")
    for g in GROUPS:
        vals = [cells[(g, s, 1.0)] for s in SEEDS if (g, s, 1.0) in cells]
        if not vals:
            continue
        corr = np.mean([c["V_metrics"]["corr"] for c in vals])
        bias = np.mean([c["V_metrics"]["bias"] for c in vals])
        rmse = np.mean([c["V_metrics"]["rmse"] for c in vals])
        ev = np.mean([c["V_metrics"]["ev"] for c in vals])
        gm = np.mean([c["G_mean"] for c in vals])
        gs = np.mean([c["G_std"] for c in vals])
        nv = int(np.mean([c["n_valid"] for c in vals]))
        lines.append(f"| {g} | {nv} | {corr:+.3f} | {bias:+.2f} | {rmse:.2f} | {ev:+.3f} | {gm:+.2f} | {gs:.2f} |")

    # Within-reward paired contrasts
    lines.append("\n## Table C1-II. PSC vs MLP V-calibration paired contrasts @ lambda=1.0\n")
    lines.append("Positive Delta corr = PSC's V better correlated with G; "
                 "negative Delta RMSE = PSC's V closer to G; "
                 "Delta EV: positive = PSC's V explains more variance.\n")
    lines.append("| Contrast | type | Delta corr | t | Delta RMSE | t | Delta EV | t |")
    lines.append("|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|")
    contrasts = [
        ("C - A", "dense", "C", "A"),
        ("F - A", "dense", "F", "A"),
        ("C - F", "dense", "C", "F"),
        ("D - B", "sparse", "D", "B"),
        ("E - B", "sparse", "E", "B"),
        ("D - E", "sparse", "D", "E"),
    ]
    for label, kind, ga, gb in contrasts:
        per_seed_diffs = {"corr": [], "rmse": [], "ev": []}
        for s in SEEDS:
            if (ga, s, 1.0) not in cells or (gb, s, 1.0) not in cells:
                continue
            ca = cells[(ga, s, 1.0)]["V_metrics"]
            cb = cells[(gb, s, 1.0)]["V_metrics"]
            per_seed_diffs["corr"].append(ca["corr"] - cb["corr"])
            per_seed_diffs["rmse"].append(ca["rmse"] - cb["rmse"])
            per_seed_diffs["ev"].append(ca["ev"] - cb["ev"])
        row = [label, kind]
        for key in ["corr", "rmse", "ev"]:
            d = np.array(per_seed_diffs[key])
            m, t, _ = paired_t(d)
            row.append(f"{m:+.3f} ({sig(t)})")
            row.append(f"{t:+.2f}")
        lines.append("| " + " | ".join(row) + " |")

    # V_phys vs V_res alone (PSC groups)
    lines.append("\n## Table C1-III. V_phys vs V_res calibration (PSC groups, lambda=1.0)\n")
    lines.append("Decomposition: how well does each component alone correlate with G?\n")
    lines.append("| Group | corr(V_phys, G) | RMSE(V_phys, G) | corr(V_res, G) | RMSE(V_res, G) | corr(V_total, G) | RMSE(V_total, G) |")
    lines.append("|:-:|:-:|:-:|:-:|:-:|:-:|:-:|")
    for g in ["C", "D", "E"]:
        vals = [cells[(g, s, 1.0)] for s in SEEDS if (g, s, 1.0) in cells]
        if not vals:
            continue
        cp = np.mean([c["Vphys_metrics"]["corr"] for c in vals])
        rp = np.mean([c["Vphys_metrics"]["rmse"] for c in vals])
        cr = np.mean([c["Vres_metrics"]["corr"] for c in vals])
        rr = np.mean([c["Vres_metrics"]["rmse"] for c in vals])
        ct = np.mean([c["V_metrics"]["corr"] for c in vals])
        rt = np.mean([c["V_metrics"]["rmse"] for c in vals])
        lines.append(f"| {g} | {cp:+.3f} | {rp:.2f} | {cr:+.3f} | {rr:.2f} | {ct:+.3f} | {rt:.2f} |")

    # Per-lambda trend
    lines.append("\n## Table C1-IV. V_theta corr(V,G) vs lambda (5-seed mean)\n")
    lines.append("| Group | lam=0.0 | lam=0.3 | lam=0.5 | lam=0.7 | lam=1.0 |")
    lines.append("|:-:|:-:|:-:|:-:|:-:|:-:|")
    for g in GROUPS:
        row = [g]
        for lam in LAMBDAS:
            vals = [cells[(g, s, lam)]["V_metrics"]["corr"] for s in SEEDS if (g, s, lam) in cells]
            if not vals:
                row.append("—")
            else:
                row.append(f"{np.mean(vals):+.3f}")
        lines.append("| " + " | ".join(row) + " |")

    with open(f"{OUT}/c1_table.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[+] wrote {OUT}/c1_table.md")

    # Figure: corr vs lambda
    fig, axes = plt.subplots(1, 2, figsize=(DBL_WIDTH, ROW_H))
    for ax, metric, ylabel in [
        (axes[0], "corr", r"corr$(V_\theta, G)$"),
        (axes[1], "ev", r"explained variance"),
    ]:
        for g in GROUPS:
            ys = []
            es = []
            for lam in LAMBDAS:
                vals = [cells[(g, s, lam)]["V_metrics"][metric] for s in SEEDS if (g, s, lam) in cells]
                if not vals:
                    ys.append(np.nan); es.append(np.nan)
                else:
                    ys.append(float(np.mean(vals))); es.append(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0)
            ax.errorbar(LAMBDAS, ys, yerr=es, label=LABELS[g], color=COLORS[g],
                        lw=1.3, capsize=2, marker='o', markersize=4)
        ax.set_xlabel(r"Test-time DR strength $\lambda$")
        ax.set_ylabel(ylabel)
        ax.set_xlim(-0.05, 1.05)
        if metric == "corr":
            ax.legend(loc="best", framealpha=0.85, edgecolor="none", fontsize=7)
    plt.tight_layout(pad=0.5)
    fig.savefig(f"{OUT}/c1_calibration_curves.pdf")
    fig.savefig(f"{OUT}/c1_calibration_curves.png", dpi=160)
    plt.close(fig)
    print(f"[+] wrote {OUT}/c1_calibration_curves.pdf")

    # Verdict
    print()
    print("=" * 76)
    print("C1 verdict — does PSC's V stay better-calibrated under OOD?")
    print("=" * 76)
    for label, kind, ga, gb in contrasts:
        diffs = {"corr": [], "rmse": [], "ev": []}
        for s in SEEDS:
            if (ga, s, 1.0) not in cells or (gb, s, 1.0) not in cells:
                continue
            ca = cells[(ga, s, 1.0)]["V_metrics"]
            cb = cells[(gb, s, 1.0)]["V_metrics"]
            for k in diffs:
                diffs[k].append(ca[k] - cb[k])
        m_corr, t_corr, _ = paired_t(np.array(diffs["corr"]))
        m_rmse, t_rmse, _ = paired_t(np.array(diffs["rmse"]))
        m_ev, t_ev, _ = paired_t(np.array(diffs["ev"]))
        print(f"  {label:7s} ({kind:6s}): "
              f"d_corr={m_corr:+.3f} ({sig(t_corr)})  "
              f"d_RMSE={m_rmse:+.2f} ({sig(t_rmse)})  "
              f"d_EV={m_ev:+.3f} ({sig(t_ev)})")


if __name__ == "__main__":
    main()
