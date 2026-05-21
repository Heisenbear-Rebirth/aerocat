"""
H5 — Per-dimension OOD perturbation decomposition analysis.

After running run_dr_eval_perdim.py to get per-(group, seed, dim) crash/return stats,
this script:
  1. Aggregates across 5 seeds → mean/SD per (group, dim)
  2. Computes "dim contribution" = crash[dim] - crash[nominal]
  3. Tests whether D's safety advantage over A holds in each dim (paired t)
  4. Identifies which dim(s) drive most of the crash gap at all_l1

Outputs:
  experiments/_h5_perdim/
    h5_table.md
    h5_crash_per_dim.{pdf,png}      bar chart per group × dim
    h5_dim_decomposition.{pdf,png}  contribution stacked bars
"""
import json
import os
from typing import Dict, List

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
SEEDS = [42, 123, 456, 789, 1024]
DIMS = ["nominal", "mass", "wind", "turb", "sensor", "actuator",
        "init_state", "collision", "all_l1"]

import argparse
_ap = argparse.ArgumentParser()
_ap.add_argument("--input", type=str, default="experiments/_h5_perdim/results.json")
_ap.add_argument("--output-dir", type=str, default="experiments/_h5_perdim")
_ap.add_argument("--groups", nargs="+", default=["A", "B", "C", "D", "E", "F"])
_ap.add_argument("--title-tag", type=str, default="H5", help="Tag for figure titles / verdict header")
_args, _ = _ap.parse_known_args()
GROUPS = _args.groups
IN = _args.input
OUT = _args.output_dir
TITLE_TAG = _args.title_tag


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
    with open(IN) as f:
        rows = json.load(f)

    # Build (group, seed, dim) → row dict
    table: Dict = {}
    for r in rows:
        if "error" in r: continue
        key = (r["group"], r["seed"], r["dim"])
        table[key] = r

    lines = []
    lines.append("# H5 — Per-Dimension OOD Perturbation Decomposition\n")
    lines.append(f"Eval: 6 groups × 5 seeds × {len(DIMS)} dims, deterministic policy, 100 episodes per cell, 256 envs.\n")
    lines.append("Dims activate λ=1 in one perturbation axis; all others stay λ=0.\n")
    lines.append(f"Sanity: nominal ≈ baseline (λ=0); all_l1 ≈ P0 λ=1.0 results.\n")

    # ----- Per-dim crash rate table -----
    lines.append("\n## (1) Crash rate per (group, dim), mean ± SD over 5 seeds\n")
    lines.append("| Group | " + " | ".join(DIMS) + " |")
    lines.append("|:---:|" + ":---:|" * len(DIMS))
    crash_mat = {}
    for g in GROUPS:
        row = [g]
        crash_mat[g] = {}
        for dim in DIMS:
            vals = [table[(g, s, dim)]["crash_rate"] for s in SEEDS if (g, s, dim) in table]
            crash_mat[g][dim] = vals
            if len(vals) < 2:
                row.append("—")
            else:
                row.append(f"{np.mean(vals)*100:.1f}±{np.std(vals, ddof=1)*100:.1f}")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("\n(Values in percent. SD across 5 seeds.)\n")

    # ----- Per-dim Δcrash = dim − nominal -----
    lines.append("\n## (2) Δcrash = dim − nominal per (group, dim) (paired by seed)\n")
    lines.append("How much does activating ONE OOD dim alone add to crash rate vs full-nominal?\n")
    lines.append("| Group | " + " | ".join([d for d in DIMS if d != "nominal"]) + " |")
    lines.append("|:---:|" + ":---:|" * (len(DIMS) - 1))
    delta_mat = {}
    for g in GROUPS:
        row = [g]
        delta_mat[g] = {}
        nom_per_seed = {s: table[(g, s, "nominal")]["crash_rate"] for s in SEEDS if (g, s, "nominal") in table}
        for dim in DIMS:
            if dim == "nominal": continue
            diffs = []
            for s in SEEDS:
                if (g, s, dim) not in table or s not in nom_per_seed: continue
                diffs.append(table[(g, s, dim)]["crash_rate"] - nom_per_seed[s])
            delta_mat[g][dim] = diffs
            if len(diffs) < 2:
                row.append("—")
            else:
                m, t, d, n = paired_t(diffs)
                sig = "**" if abs(t) > 2.776 else ("·" if abs(t) > 2.132 else "")
                row.append(f"{m*100:+.1f}{sig}")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("\n** = p<0.05 (|t|>2.776, df=4); · = p<0.10. Percent units.\n")

    # ----- Sanity: nominal vs P0 λ=0; all_l1 vs P0 λ=1 -----
    lines.append("\n## (3) Sanity check: nominal vs all_l1 (no P0 cross-ref here; manually compare)\n")
    lines.append("| Group | nominal crash | all_l1 crash | Δ |")
    lines.append("|:---:|:---:|:---:|:---:|")
    for g in GROUPS:
        nom = crash_mat[g].get("nominal", [])
        all1 = crash_mat[g].get("all_l1", [])
        if len(nom) >= 2 and len(all1) >= 2:
            lines.append(f"| {g} | {np.mean(nom)*100:.1f}% | {np.mean(all1)*100:.1f}% | "
                         f"{(np.mean(all1)-np.mean(nom))*100:+.1f}% |")

    # ----- Within-group: which dim drives the largest Δcrash? -----
    lines.append("\n## (4) Per-group ranking — which dim adds the most crash?\n")
    lines.append("| Group | Top-1 dim (Δcrash) | Top-2 | Top-3 |")
    lines.append("|:---:|:---:|:---:|:---:|")
    for g in GROUPS:
        ranked = []
        for dim in DIMS:
            if dim in ("nominal", "all_l1"): continue
            vals = delta_mat[g].get(dim, [])
            if len(vals) < 2: continue
            ranked.append((dim, np.mean(vals)))
        ranked.sort(key=lambda x: -x[1])
        top3 = ranked[:3]
        cells = [f"{d} ({v*100:+.1f}%)" for d, v in top3]
        cells += ["—"] * (3 - len(top3))
        lines.append(f"| {g} | " + " | ".join(cells) + " |")

    # ----- Cross-group contrasts at each dim: does D have advantage over A in this dim? -----
    lines.append("\n## (5) D − A crash gap per dim (paired by seed)\n")
    lines.append("Tests whether PSC+sparse's OOD-safety advantage over MLP+dense holds dim-by-dim.\n")
    lines.append("Negative ⇒ D safer than A in that dim.\n")
    lines.append("| Dim | D − A Δcrash | t (df=4) | n | Direction |")
    lines.append("|:---:|:---:|:---:|:---:|:---:|")
    for dim in DIMS:
        diffs = []
        for s in SEEDS:
            d_rec = table.get(("D", s, dim))
            a_rec = table.get(("A", s, dim))
            if d_rec and a_rec:
                diffs.append(d_rec["crash_rate"] - a_rec["crash_rate"])
        m, t, d, n = paired_t(diffs)
        if np.isnan(m):
            lines.append(f"| {dim} | — | — | {n} | — |")
            continue
        if abs(t) > 2.776:
            verd = "↓ D safer" if m < 0 else "↑ A safer"
        elif abs(t) > 2.132:
            verd = f"{'↓' if m < 0 else '↑'} (p<0.10)"
        else:
            verd = "n.s."
        lines.append(f"| {dim} | {m*100:+.1f}% | {t:+.2f} | {n} | {verd} |")

    # ----- Tracking RMSE per dim -----
    lines.append("\n## (6) Tracking RMSE per (group, dim), mean over 5 seeds\n")
    lines.append("Captures non-crash failure modes (e.g. mass changes hurt tracking but not safety).\n")
    lines.append("| Group | " + " | ".join(DIMS) + " |")
    lines.append("|:---:|" + ":---:|" * len(DIMS))
    for g in GROUPS:
        row = [g]
        for dim in DIMS:
            vals = [table[(g, s, dim)]["tracking_rmse"] for s in SEEDS if (g, s, dim) in table]
            if len(vals) < 2:
                row.append("—")
            else:
                row.append(f"{np.mean(vals):.2f}")
        lines.append("| " + " | ".join(row) + " |")

    # ----- Episode length per (group, dim) -----
    lines.append("\n## (7) Mean episode length (steps) per (group, dim)\n")
    lines.append("CRITICAL CAVEAT: per-episode crash rate confounds with exposure time. Dense (A/C/F)\n")
    lines.append("episodes run to ~500 steps (timeout); sparse (B/D/E) terminate at goal in 5-50 steps.\n")
    lines.append("Section (8) reports per-second crash hazard for fair safety comparison.\n")
    lines.append("| Group | " + " | ".join(DIMS) + " |")
    lines.append("|:---:|" + ":---:|" * len(DIMS))
    eplen_mat = {}
    for g in GROUPS:
        row = [g]
        eplen_mat[g] = {}
        for dim in DIMS:
            vals = [table[(g, s, dim)]["mean_episode_length"] for s in SEEDS if (g, s, dim) in table]
            eplen_mat[g][dim] = vals
            if len(vals) < 2:
                row.append("—")
            else:
                row.append(f"{np.mean(vals):.0f}")
        lines.append("| " + " | ".join(row) + " |")

    # ----- Per-second crash hazard rate -----
    # hazard = crash_rate / (mean_ep_len * dt). dt = 0.02s. Units: failures per second of OOD exposure.
    DT = 0.02
    lines.append(f"\n## (8) Per-second crash hazard rate (1/s) per (group, dim), mean over 5 seeds\n")
    lines.append(f"hazard = crash_rate / (mean_ep_length × {DT}s).\n")
    lines.append("Removes exposure-time confound: 'given 1 second of OOD flight, P(crash)'.\n")
    lines.append("Use for fair safety comparison across groups with different episode termination patterns.\n")
    lines.append("| Group | " + " | ".join(DIMS) + " |")
    lines.append("|:---:|" + ":---:|" * len(DIMS))
    hazard_mat = {}
    for g in GROUPS:
        row = [g]
        hazard_mat[g] = {}
        for dim in DIMS:
            vals = []
            for s in SEEDS:
                if (g, s, dim) not in table: continue
                c = table[(g, s, dim)]["crash_rate"]
                l = table[(g, s, dim)]["mean_episode_length"]
                if l > 0:
                    vals.append(c / (l * DT))
            hazard_mat[g][dim] = vals
            if len(vals) < 2:
                row.append("—")
            else:
                row.append(f"{np.mean(vals):.3f}")
        lines.append("| " + " | ".join(row) + " |")

    # ----- Hazard-based D-A contrasts -----
    lines.append(f"\n## (9) D − A hazard rate gap per dim (paired by seed)\n")
    lines.append("Repeats (5) but using per-second hazard. Negative ⇒ D's per-step robustness > A's.\n")
    lines.append("| Dim | D − A Δhazard (/s) | t (df=4) | Direction |")
    lines.append("|:---:|:---:|:---:|:---:|")
    for dim in DIMS:
        diffs = []
        for s in SEEDS:
            d_rec = table.get(("D", s, dim)); a_rec = table.get(("A", s, dim))
            if d_rec and a_rec:
                d_l = d_rec["mean_episode_length"]; a_l = a_rec["mean_episode_length"]
                if d_l > 0 and a_l > 0:
                    d_h = d_rec["crash_rate"] / (d_l * DT)
                    a_h = a_rec["crash_rate"] / (a_l * DT)
                    diffs.append(d_h - a_h)
        m, t, d, n = paired_t(diffs)
        if np.isnan(m):
            lines.append(f"| {dim} | — | — | — |"); continue
        if abs(t) > 2.776:
            verd = "↓ D per-step safer" if m < 0 else "↑ A per-step safer"
        elif abs(t) > 2.132:
            verd = f"{'↓' if m < 0 else '↑'} (p<0.10)"
        else:
            verd = "n.s."
        lines.append(f"| {dim} | {m:+.3f} | {t:+.2f} | {verd} |")

    out_basename = TITLE_TAG.lower()
    with open(f"{OUT}/{out_basename}_table.md", "w") as f:
        f.write("\n".join(lines))
    print(f"[+] wrote {OUT}/{out_basename}_table.md")

    # ============================================================
    # Figures
    # ============================================================
    # Fig 1: crash rate per (group, dim) heatmap-style bars
    fig, axes = plt.subplots(2, 3, figsize=(DBL_WIDTH * 1.0, ROW_H * 2.0), sharey=True)
    axes = axes.flatten()
    pert_dims = [d for d in DIMS if d not in ("nominal", "all_l1")]
    for i, g in enumerate(GROUPS):
        ax = axes[i]
        means = []
        sds = []
        labels = []
        for dim in DIMS:
            vals = crash_mat[g].get(dim, [])
            if len(vals) < 2:
                means.append(0); sds.append(0); labels.append(dim); continue
            means.append(np.mean(vals) * 100)
            sds.append(np.std(vals, ddof=1) * 100)
            labels.append(dim)
        bars = ax.bar(range(len(labels)), means, yerr=sds, capsize=2,
                      color=[COLORS[g] if d not in ("nominal", "all_l1") else "#cccccc" for d in labels],
                      edgecolor='black', linewidth=0.4)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
        ax.set_ylabel("Crash %")
        ax.set_title(LABELS[g], fontsize=9)
    plt.tight_layout(pad=0.5)
    fig.savefig(f"{OUT}/{out_basename}_crash_per_dim.pdf"); fig.savefig(f"{OUT}/{out_basename}_crash_per_dim.png", dpi=160)
    plt.close(fig)
    print(f"[+] wrote {OUT}/{out_basename}_crash_per_dim.pdf")

    # Fig 2: Δcrash per dim, grouped bars (all groups side by side)
    fig, ax = plt.subplots(1, 1, figsize=(DBL_WIDTH, ROW_H * 1.2))
    width = 0.13
    x = np.arange(len(pert_dims))
    for gi, g in enumerate(GROUPS):
        means = []
        sds = []
        for dim in pert_dims:
            vals = delta_mat[g].get(dim, [])
            if len(vals) < 2: means.append(0); sds.append(0); continue
            means.append(np.mean(vals) * 100)
            sds.append(np.std(vals, ddof=1) * 100)
        ax.bar(x + gi * width - 2.5 * width, means, width, yerr=sds,
               color=COLORS[g], label=LABELS[g], edgecolor='black', linewidth=0.4, capsize=2)
    ax.set_xticks(x)
    ax.set_xticklabels(pert_dims)
    ax.set_ylabel("Δ crash rate vs nominal (%)")
    ax.set_title("OOD perturbation decomposition: which dim adds most crash?", fontsize=10)
    ax.legend(loc="best", framealpha=0.85, edgecolor='none', fontsize=7, ncol=3)
    plt.tight_layout(pad=0.5)
    fig.savefig(f"{OUT}/{out_basename}_dim_decomposition.pdf"); fig.savefig(f"{OUT}/{out_basename}_dim_decomposition.png", dpi=160)
    plt.close(fig)
    print(f"[+] wrote {OUT}/{out_basename}_dim_decomposition.pdf")

    # ============================================================
    # Verdict
    # ============================================================
    print()
    print("=" * 78)
    print(f"{TITLE_TAG} verdict — which OOD dim drives crash?")
    print("=" * 78)
    print(f"\n{'Group':>6}  {'nominal':>8}  {'all_l1':>7}  | top-3 dims by Δcrash")
    for g in GROUPS:
        nom = crash_mat[g].get("nominal", [])
        all1 = crash_mat[g].get("all_l1", [])
        nom_m = np.mean(nom) * 100 if len(nom) >= 2 else float("nan")
        all_m = np.mean(all1) * 100 if len(all1) >= 2 else float("nan")
        ranked = []
        for dim in DIMS:
            if dim in ("nominal", "all_l1"): continue
            vals = delta_mat[g].get(dim, [])
            if len(vals) < 2: continue
            ranked.append((dim, np.mean(vals) * 100))
        ranked.sort(key=lambda x: -x[1])
        top3_str = "  ".join([f"{d}:{v:+.1f}%" for d, v in ranked[:3]])
        print(f"{g:>6}  {nom_m:>7.1f}%  {all_m:>6.1f}%  | {top3_str}")
    print()
    print("D − A gap per dim:")
    for dim in DIMS:
        diffs = []
        for s in SEEDS:
            d_rec = table.get(("D", s, dim))
            a_rec = table.get(("A", s, dim))
            if d_rec and a_rec:
                diffs.append(d_rec["crash_rate"] - a_rec["crash_rate"])
        m, t, _, n = paired_t(diffs)
        if not np.isnan(m):
            print(f"  {dim:>11s}: Δ={m*100:+.2f}%  t={t:+.2f}  n={n}")
    print()


if __name__ == "__main__":
    main()
