"""
H1 — Deployment-time V_phys ratio evolution under OOD lambda sweep.

A2 measured the v_phys_ratio during TRAINING (5-50M env steps) and found PSC groups
have V_phys carrying 65-100% of V early. H1 measures the analogue at DEPLOYMENT time
across the OOD lambda sweep, using the C1 calibration trajectories.

Hypothesis: As lambda increases (OOD severity rises), V_res — which depends on the
training distribution — fails, while V_phys — analytic in state — remains. So the
ratio |V_phys|/(|V_phys|+|V_res|) should rise with lambda for PSC groups (C/D/E).

Per-cell metrics (mean over valid/active steps, all 256 envs):
  ratio_abs = mean(|v_phys|) / (mean(|v_phys|) + mean(|v_res|))   primary
  vphys_mag = mean(|v_phys|)
  vres_mag  = mean(|v_res|)
  v_mag     = mean(|value|)
  calib_abs_err = mean(|value - G|)   from reconstructed G (same as C1)
  corr_ratio_err: per-env Pearson(ratio_t, |V_t - G_t|), median across envs

Key contrasts (paired across 5 seeds):
  PSC groups only: ratio[lam=1.0] - ratio[lam=0.0] per group (C/D/E)
  Direction: positive (ratio rises with OOD) supports "V_phys takes over" hypothesis

Output:
  experiments/_h1_vphys_evolution/
    h1_table.md
    h1_ratio_vs_lambda.{pdf,png}      ratio for PSC groups, line plot
    h1_components_vs_lambda.{pdf,png} |v_phys| and |v_res| separately
    h1_per_seed_ratios.json           raw per-cell values
"""
import glob
import json
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
PSC_GROUPS = ["C", "D", "E"]
ALL_GROUPS = ["A", "B", "C", "D", "E", "F"]
SEEDS = [42, 123, 456, 789, 1024]
LAMBDAS = [0.0, 0.3, 0.5, 0.7, 1.0]
GAMMA = 0.995

TRAJ_DIR = "experiments/_c1/T1_calib_trajectories"
OUT = "experiments/_h1_vphys_evolution"


def reconstruct_G(reward: np.ndarray, done: np.ndarray, gamma: float = GAMMA):
    T, E = reward.shape
    G = np.zeros((T, E), dtype=np.float64)
    valid = np.zeros((T, E), dtype=bool)
    for e in range(E):
        idxs = np.where(done[:, e] == 1)[0]
        if len(idxs) == 0:
            continue
        last_done = int(idxs[-1])
        G[last_done, e] = reward[last_done, e]
        valid[last_done, e] = True
        for t in range(last_done - 1, -1, -1):
            if done[t, e] == 1:
                G[t, e] = reward[t, e]
            else:
                G[t, e] = reward[t, e] + gamma * G[t + 1, e]
            valid[t, e] = True
    return G, valid


def process_cell(path: str) -> dict:
    z = np.load(path)
    v = z["value"].astype(np.float64)
    vp = z["v_phys"].astype(np.float64)
    vr = z["v_res"].astype(np.float64)
    reward = z["reward"].astype(np.float64)
    done = z["done"]
    active = z["active"].astype(bool)
    G, valid = reconstruct_G(reward, done)
    # Use intersection of active and valid for ratio
    mask = active & valid
    if not mask.any():
        # fall back to active only (ratio is well-defined even without G)
        mask_for_mag = active
    else:
        mask_for_mag = mask

    vpa = np.abs(vp[mask_for_mag])
    vra = np.abs(vr[mask_for_mag])
    va = np.abs(v[mask_for_mag])

    if vpa.size == 0:
        ratio = np.nan; vphys_mag = np.nan; vres_mag = np.nan; v_mag = np.nan
    else:
        vphys_mag = float(vpa.mean())
        vres_mag = float(vra.mean())
        v_mag = float(va.mean())
        denom = vphys_mag + vres_mag
        ratio = vphys_mag / denom if denom > 0 else np.nan

    # Calibration metrics on the same mask
    if mask.any():
        Vm = v[mask]
        Gm = G[mask]
        calib_abs_err = float(np.mean(np.abs(Vm - Gm)))
    else:
        calib_abs_err = np.nan

    return dict(
        ratio=ratio, vphys_mag=vphys_mag, vres_mag=vres_mag, v_mag=v_mag,
        calib_abs_err=calib_abs_err,
        n_active=int(active.sum()), n_valid=int(mask.sum()),
    )


def paired_t(diffs):
    a = np.array([d for d in diffs if not np.isnan(d)])
    n = len(a)
    if n < 2: return float("nan"), float("nan"), float("nan"), n
    m = float(a.mean())
    sd = float(a.std(ddof=1))
    if sd == 0: return m, float("inf"), float("inf"), n
    t = m / (sd / np.sqrt(n))
    d = m / sd
    return m, t, d, n


def main():
    os.makedirs(OUT, exist_ok=True)

    # ---- Collect per-cell metrics ----
    raw: Dict[Tuple[str, int, float], dict] = {}
    for g in ALL_GROUPS:
        for s in SEEDS:
            for lam in LAMBDAS:
                fn = f"{TRAJ_DIR}/traj_{g}_{s}_lam{lam}.npz"
                if not os.path.exists(fn):
                    raw[(g, s, lam)] = None
                    continue
                raw[(g, s, lam)] = process_cell(fn)

    # ---- Per-cell JSON dump ----
    dump = {}
    for (g, s, lam), v in raw.items():
        if v is None: continue
        dump[f"{g}_{s}_lam{lam}"] = v
    with open(f"{OUT}/h1_per_seed_ratios.json", "w") as f:
        json.dump(dump, f, indent=2)
    print(f"[+] wrote {OUT}/h1_per_seed_ratios.json ({len(dump)} cells)")

    # ---- Table: ratio per (group, lambda), mean ± std over 5 seeds ----
    lines = []
    lines.append("# H1 — Deployment-time $|V_{\\mathrm{phys}}|/(|V_{\\mathrm{phys}}|+|V_{\\mathrm{res}}|)$ vs OOD $\\lambda$\n")
    lines.append("Tests whether PSC's V_phys takes over as V_res fails under OOD.\n")
    lines.append("Data: C1 T1_calib_trajectories, 6 groups × 5 seeds × 5 lambdas = 150 cells.\n")
    lines.append("Ratio uses mean(|V_phys|) and mean(|V_res|) over active+valid steps.\n")

    lines.append("\n## (1) Mean ratio per (group, lambda), 5 seeds\n")
    lines.append("MLP-only groups (A/B/F) have V_phys=0 by construction → ratio=0 (omitted).\n")
    lines.append("| Group | λ=0.0 | λ=0.3 | λ=0.5 | λ=0.7 | λ=1.0 |")
    lines.append("|:---:|:---:|:---:|:---:|:---:|:---:|")
    cell_table = {}
    for g in PSC_GROUPS:
        row = [g]
        cell_table[g] = {}
        for lam in LAMBDAS:
            vals = [raw[(g, s, lam)]["ratio"] for s in SEEDS
                    if raw[(g, s, lam)] is not None and not np.isnan(raw[(g, s, lam)]["ratio"])]
            cell_table[g][lam] = vals
            if len(vals) < 2:
                row.append("—")
            else:
                row.append(f"{np.mean(vals):.3f} ± {np.std(vals, ddof=1):.3f}")
        lines.append("| " + " | ".join(row) + " |")

    # ---- Key contrast: ratio[λ=1.0] - ratio[λ=0.0] per group, paired 5 seeds ----
    lines.append("\n## (2) Paired Δratio = ratio[λ=1.0] − ratio[λ=0.0] (5 seeds)\n")
    lines.append("Positive ⇒ V_phys takes over under OOD (V_res fails). Negative ⇒ V_res still dominates (or grows).\n")
    lines.append("| Group | Mean Δratio | t (df=4) | Cohen's d | n | Direction |")
    lines.append("|:---:|:---:|:---:|:---:|:---:|:---:|")
    contrast_results = {}
    for g in PSC_GROUPS:
        diffs = []
        for s in SEEDS:
            r0 = raw[(g, s, 0.0)]; r1 = raw[(g, s, 1.0)]
            if r0 is None or r1 is None: continue
            if np.isnan(r0["ratio"]) or np.isnan(r1["ratio"]): continue
            diffs.append(r1["ratio"] - r0["ratio"])
        m, t, d, n = paired_t(diffs)
        contrast_results[g] = dict(m=m, t=t, d=d, n=n)
        # 2-sided t critical at df=4 (5 seeds): t_0.05=2.776, t_0.10=2.132
        if np.isnan(m):
            verd = "—"
        elif abs(t) > 2.776:
            verd = "↑ V_phys takes over" if m > 0 else "↓ V_res grows"
        elif abs(t) > 2.132:
            verd = "↑ (p<0.10)" if m > 0 else "↓ (p<0.10)"
        else:
            verd = "n.s."
        lines.append(f"| {g} | {m:+.3f} | {t:+.2f} | {d:+.2f} | {n} | {verd} |")

    # ---- Component decomposition: |V_phys| vs |V_res| separately across λ ----
    lines.append("\n## (3) Component magnitudes (mean ± std over 5 seeds)\n")
    lines.append("If ratio rises with λ, decompose: is it because |V_phys| rises, |V_res| falls, or both?\n")
    for g in PSC_GROUPS:
        lines.append(f"\n### Group {g}\n")
        lines.append("| Quantity | λ=0.0 | λ=0.3 | λ=0.5 | λ=0.7 | λ=1.0 |")
        lines.append("|:---:|:---:|:---:|:---:|:---:|:---:|")
        for key, label in [("vphys_mag", "|V_phys|"), ("vres_mag", "|V_res|"),
                           ("v_mag", "|V|"), ("calib_abs_err", "|V−G|")]:
            row = [label]
            for lam in LAMBDAS:
                vals = [raw[(g, s, lam)][key] for s in SEEDS
                        if raw[(g, s, lam)] is not None and not np.isnan(raw[(g, s, lam)][key])]
                if len(vals) < 2:
                    row.append("—")
                else:
                    row.append(f"{np.mean(vals):.2f}±{np.std(vals, ddof=1):.2f}")
            lines.append("| " + " | ".join(row) + " |")

    # ---- Cross-group sanity: do MLP groups (A/B/F) calibration errors also rise? ----
    lines.append("\n## (4) Calibration error |V−G| across λ (all groups, mean only)\n")
    lines.append("If PSC stays calibrated while MLP degrades, that's deployment-time V calibration evidence.\n")
    lines.append("| Group | λ=0.0 | λ=0.3 | λ=0.5 | λ=0.7 | λ=1.0 |")
    lines.append("|:---:|:---:|:---:|:---:|:---:|:---:|")
    for g in ALL_GROUPS:
        row = [g]
        for lam in LAMBDAS:
            vals = [raw[(g, s, lam)]["calib_abs_err"] for s in SEEDS
                    if raw[(g, s, lam)] is not None and not np.isnan(raw[(g, s, lam)]["calib_abs_err"])]
            if len(vals) < 2:
                row.append("—")
            else:
                row.append(f"{np.mean(vals):.2f}")
        lines.append("| " + " | ".join(row) + " |")

    # ---- Δ|V−G| per group with paired t-stat ----
    lines.append("\n## (5) Paired Δ|V−G| = err[λ=1.0] − err[λ=0.0] per group (5 seeds)\n")
    lines.append("Negative = calibration **improves** under OOD (counter-intuitive). Positive = degrades (expected).\n")
    lines.append("| Group | Mean Δ|V−G| | t (df=4) | d | n | Direction |")
    lines.append("|:---:|:---:|:---:|:---:|:---:|:---:|")
    calib_deltas = {}
    for g in ALL_GROUPS:
        diffs = []
        for s in SEEDS:
            r0 = raw[(g, s, 0.0)]; r1 = raw[(g, s, 1.0)]
            if r0 is None or r1 is None: continue
            if np.isnan(r0["calib_abs_err"]) or np.isnan(r1["calib_abs_err"]): continue
            diffs.append(r1["calib_abs_err"] - r0["calib_abs_err"])
        m, t, d, n = paired_t(diffs)
        calib_deltas[g] = dict(diffs=diffs, m=m, t=t, d=d, n=n)
        if np.isnan(m):
            verd = "—"
        elif abs(t) > 2.776:
            verd = "↑ degrades" if m > 0 else "↓ improves"
        elif abs(t) > 2.132:
            verd = f"{'↑' if m > 0 else '↓'} (p<0.10)"
        else:
            verd = "n.s."
        lines.append(f"| {g} | {m:+.2f} | {t:+.2f} | {d:+.2f} | {n} | {verd} |")

    # ---- Cross-group paired contrasts on Δ|V−G| for dense reward (A/C/F) ----
    lines.append("\n## (6) Dense-reward calibration: paired contrast on Δ|V−G| (A vs C vs F)\n")
    lines.append("Tests whether PSC (C) keeps |V−G| lower under OOD than MLP (A) or Cai dual (F).\n")
    lines.append("Each row pairs by seed and compares one group's Δ|V−G| to another's.\n")
    lines.append("| Contrast | Mean diff | t (df=4) | d | n | Direction |")
    lines.append("|:---:|:---:|:---:|:---:|:---:|:---:|")
    dense_contrasts = [("C - A", "C", "A"), ("F - A", "F", "A"), ("C - F", "C", "F")]
    for label, ga, gb in dense_contrasts:
        da = calib_deltas.get(ga, {}).get("diffs", [])
        db = calib_deltas.get(gb, {}).get("diffs", [])
        if len(da) != len(db) or len(da) < 2: continue
        diffs = [a - b for a, b in zip(da, db)]
        m, t, d, n = paired_t(diffs)
        if np.isnan(m): continue
        if abs(t) > 2.776:
            verd = f"{ga} more robust" if m < 0 else f"{gb} more robust"
        elif abs(t) > 2.132:
            verd = "trend (p<0.10)"
        else:
            verd = "n.s."
        lines.append(f"| {label} | {m:+.2f} | {t:+.2f} | {d:+.2f} | {n} | {verd} |")

    with open(f"{OUT}/h1_table.md", "w") as f:
        f.write("\n".join(lines))
    print(f"[+] wrote {OUT}/h1_table.md")

    # ============================================================
    # Figures
    # ============================================================
    # Fig 1: ratio vs lambda for PSC groups
    fig, ax = plt.subplots(1, 1, figsize=(COL_WIDTH, ROW_H))
    for g in PSC_GROUPS:
        mus, q1s, q3s = [], [], []
        for lam in LAMBDAS:
            vals = cell_table[g][lam]
            if len(vals) < 2: mus.append(np.nan); q1s.append(np.nan); q3s.append(np.nan); continue
            mus.append(np.mean(vals))
            q1s.append(np.percentile(vals, 25))
            q3s.append(np.percentile(vals, 75))
        ax.plot(LAMBDAS, mus, "-o", color=COLORS[g], label=LABELS[g], lw=1.5, ms=4)
        ax.fill_between(LAMBDAS, q1s, q3s, color=COLORS[g], alpha=0.15, lw=0)
    ax.set_xlabel("OOD severity λ")
    ax.set_ylabel("$|V_{\\mathrm{phys}}|/(|V_{\\mathrm{phys}}|+|V_{\\mathrm{res}}|)$")
    ax.set_title("Deployment-time V_phys take-over under OOD")
    ax.legend(loc="best", framealpha=0.85, edgecolor='none', fontsize=8)
    ax.set_ylim(0, max(0.6, ax.get_ylim()[1]))
    plt.tight_layout(pad=0.5)
    fig.savefig(f"{OUT}/h1_ratio_vs_lambda.pdf"); fig.savefig(f"{OUT}/h1_ratio_vs_lambda.png", dpi=160)
    plt.close(fig)
    print(f"[+] wrote {OUT}/h1_ratio_vs_lambda.pdf")

    # Fig 2: |V_phys| and |V_res| separately across λ for PSC groups
    fig, axes = plt.subplots(1, 2, figsize=(DBL_WIDTH, ROW_H))
    for ax, key, ylabel in zip(axes,
                                ["vphys_mag", "vres_mag"],
                                ["mean $|V_{\\mathrm{phys}}|$", "mean $|V_{\\mathrm{res}}|$"]):
        for g in PSC_GROUPS:
            mus = []
            for lam in LAMBDAS:
                vals = [raw[(g, s, lam)][key] for s in SEEDS
                        if raw[(g, s, lam)] is not None and not np.isnan(raw[(g, s, lam)][key])]
                mus.append(np.mean(vals) if len(vals) >= 2 else np.nan)
            ax.plot(LAMBDAS, mus, "-o", color=COLORS[g], label=LABELS[g], lw=1.5, ms=4)
        ax.set_xlabel("OOD severity λ")
        ax.set_ylabel(ylabel)
        ax.legend(loc="best", framealpha=0.85, edgecolor='none', fontsize=7)
    axes[0].set_title("$|V_{\\mathrm{phys}}|$ vs $\\lambda$")
    axes[1].set_title("$|V_{\\mathrm{res}}|$ vs $\\lambda$")
    plt.tight_layout(pad=0.5)
    fig.savefig(f"{OUT}/h1_components_vs_lambda.pdf"); fig.savefig(f"{OUT}/h1_components_vs_lambda.png", dpi=160)
    plt.close(fig)
    print(f"[+] wrote {OUT}/h1_components_vs_lambda.pdf")

    # ============================================================
    # Verdict
    # ============================================================
    print()
    print("=" * 72)
    print("H1 verdict — does V_phys take over as V_res fails under OOD?")
    print("=" * 72)
    print(f"\n  Ratio = mean(|V_phys|) / (mean(|V_phys|) + mean(|V_res|))")
    print(f"  PSC groups: C/D/E (MLP groups have V_phys=0 by construction)\n")
    for g in PSC_GROUPS:
        r0 = cell_table[g][0.0]; r1 = cell_table[g][1.0]
        if len(r0) >= 2 and len(r1) >= 2:
            print(f"  {g}: ratio[λ=0]={np.mean(r0):.3f}  ratio[λ=1]={np.mean(r1):.3f}", end="")
            cr = contrast_results.get(g, {})
            if not np.isnan(cr.get("m", np.nan)):
                print(f"   Δ={cr['m']:+.3f}  t={cr['t']:+.2f}  d={cr['d']:+.2f}  n={cr['n']}")
            else:
                print()
    print()


if __name__ == "__main__":
    main()
