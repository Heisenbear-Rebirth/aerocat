"""
C3 — Stochastic-policy OOD safety validation.

Tests whether the §V-C C1 finding (sparse-trained policies crash 58-70% less
under extreme OOD) — established at DETERMINISTIC policy — survives when the
policy is evaluated STOCHASTICALLY (action = tanh(mean + std*eps)), i.e. under
exploration-noise-level perturbation. Directly addresses the §VI-D limitation.

Inputs:
  deterministic baseline : experiments/_dr_eval_results.json   (P0 T1, 150 cells)
  stochastic             : experiments/_c3/T1_stoch_results.json (this run)

Both share schema: list of {group, seed, lambda, crash_rate, success_rate, ...}.

Key questions:
  1. Per-group crash_rate @ λ=1.0: deterministic vs stochastic
  2. Sparse-vs-dense crash gap (D-A, B-A, E-A) retention under stochastic
  3. Does the "all sparse < all dense" ordering at λ=1.0 still hold?

Output: experiments/_c3_stochastic/
  c3_table.md
  c3_crash_det_vs_stoch.{pdf,png}   — per-group crash vs λ, det (solid) vs stoch (dashed)
"""
import json
import os
from collections import defaultdict
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

DET_PATH = "experiments/_dr_eval_results.json"
STOCH_PATH = "experiments/_c3/T1_stoch_results.json"
OUT = "experiments/_c3_stochastic"


def load(path) -> Dict[Tuple[str, int, float], float]:
    """Return {(group, seed, lambda): crash_rate}."""
    with open(path) as f:
        rows = json.load(f)
    out = {}
    for r in rows:
        if "crash_rate" not in r:
            continue
        out[(r["group"], int(r["seed"]), float(r["lambda"]))] = float(r["crash_rate"])
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
    det = load(DET_PATH)
    sto = load(STOCH_PATH)

    lines = []
    lines.append("# C3 — Stochastic-Policy OOD Safety Validation\n")
    lines.append("Deterministic baseline = `_dr_eval_results.json` (P0).  "
                 "Stochastic = `_c3/T1_stoch_results.json`.\n")
    lines.append("Action: deterministic `tanh(mean)` vs stochastic `tanh(mean+std*eps)`.\n")

    # Table 1: per-group crash @ lambda=1.0, det vs stoch
    lines.append("\n## Table C3-I. Crash rate @ λ=1.0 (5-seed mean ± SD)\n")
    lines.append("| Group | deterministic | stochastic | Δ (stoch−det) |")
    lines.append("|:-:|:-:|:-:|:-:|")
    crash10 = {}
    for g in GROUPS:
        dv = np.array([det[(g, s, 1.0)] for s in SEEDS if (g, s, 1.0) in det])
        sv = np.array([sto[(g, s, 1.0)] for s in SEEDS if (g, s, 1.0) in sto])
        crash10[g] = (dv, sv)
        if len(dv) == 0 or len(sv) == 0:
            lines.append(f"| {g} | — | — | — |")
            continue
        lines.append(f"| {g} | {dv.mean():.3f} ± {dv.std(ddof=1):.3f} "
                     f"| {sv.mean():.3f} ± {sv.std(ddof=1):.3f} "
                     f"| {sv.mean()-dv.mean():+.3f} |")

    # Table 2: sparse-vs-dense gap retention @ lambda=1.0
    lines.append("\n## Table C3-II. Sparse-vs-dense crash gap @ λ=1.0: det vs stoch\n")
    lines.append("Gap = dense − sparse (positive = sparse safer). Retention = stoch_gap / det_gap.\n")
    lines.append("| Contrast | det gap | stoch gap | retention | det t | stoch t |")
    lines.append("|:-:|:-:|:-:|:-:|:-:|:-:|")
    for (gd, gs) in [("A", "D"), ("A", "B"), ("A", "E"),
                     ("C", "D"), ("F", "D"), ("C", "B"), ("F", "B")]:
        det_diffs, sto_diffs = [], []
        for s in SEEDS:
            if (gd, s, 1.0) in det and (gs, s, 1.0) in det:
                det_diffs.append(det[(gd, s, 1.0)] - det[(gs, s, 1.0)])
            if (gd, s, 1.0) in sto and (gs, s, 1.0) in sto:
                sto_diffs.append(sto[(gd, s, 1.0)] - sto[(gs, s, 1.0)])
        det_diffs = np.array(det_diffs)
        sto_diffs = np.array(sto_diffs)
        if len(det_diffs) < 2 or len(sto_diffs) < 2:
            continue
        dm, dt, dd = paired_t(det_diffs)
        sm, st, sd_ = paired_t(sto_diffs)
        ret = (sm / dm) if dm != 0 else float("nan")
        lines.append(f"| {gd}−{gs} | {dm:+.3f} ({sig(dt)}) | {sm:+.3f} ({sig(st)}) "
                     f"| {ret:.2f}× | {dt:+.2f} | {st:+.2f} |")

    # Table 3: ordering check at lambda=1.0
    lines.append("\n## Table C3-III. 'All sparse < all dense' ordering @ λ=1.0\n")
    lines.append("| Policy | max sparse (B/D/E) | min dense (A/C/F) | clean separation? |")
    lines.append("|:-:|:-:|:-:|:-:|")
    for tag, dd in [("deterministic", det), ("stochastic", sto)]:
        sparse_means = []
        dense_means = []
        for g in ["B", "D", "E"]:
            v = [dd[(g, s, 1.0)] for s in SEEDS if (g, s, 1.0) in dd]
            if v:
                sparse_means.append(np.mean(v))
        for g in ["A", "C", "F"]:
            v = [dd[(g, s, 1.0)] for s in SEEDS if (g, s, 1.0) in dd]
            if v:
                dense_means.append(np.mean(v))
        mx_sp = max(sparse_means) if sparse_means else float("nan")
        mn_de = min(dense_means) if dense_means else float("nan")
        clean = "YES" if mx_sp < mn_de else "NO"
        lines.append(f"| {tag} | {mx_sp:.3f} | {mn_de:.3f} | {clean} |")

    # Table 4: per-lambda crash for D and A (full sweep)
    lines.append("\n## Table C3-IV. Crash rate vs λ — A & D, det vs stoch (5-seed mean)\n")
    lines.append("| λ | A det | A stoch | D det | D stoch |")
    lines.append("|:-:|:-:|:-:|:-:|:-:|")
    for lam in LAMBDAS:
        ad = np.mean([det[("A", s, lam)] for s in SEEDS if ("A", s, lam) in det])
        as_ = np.mean([sto[("A", s, lam)] for s in SEEDS if ("A", s, lam) in sto])
        dd = np.mean([det[("D", s, lam)] for s in SEEDS if ("D", s, lam) in det])
        ds = np.mean([sto[("D", s, lam)] for s in SEEDS if ("D", s, lam) in sto])
        lines.append(f"| {lam:.1f} | {ad:.3f} | {as_:.3f} | {dd:.3f} | {ds:.3f} |")

    with open(f"{OUT}/c3_table.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[+] wrote {OUT}/c3_table.md")

    # Figure: crash vs lambda, det solid + stoch dashed
    fig, ax = plt.subplots(1, 1, figsize=(COL_WIDTH * 1.4, ROW_H))
    for g in GROUPS:
        det_curve = [np.mean([det[(g, s, lam)] for s in SEEDS if (g, s, lam) in det]) for lam in LAMBDAS]
        sto_curve = [np.mean([sto[(g, s, lam)] for s in SEEDS if (g, s, lam) in sto]) for lam in LAMBDAS]
        ax.plot(LAMBDAS, det_curve, color=COLORS[g], lw=1.4, marker='o', markersize=3,
                label=f"{LABELS[g]} (det)")
        ax.plot(LAMBDAS, sto_curve, color=COLORS[g], lw=1.2, ls='--', marker='s', markersize=3,
                alpha=0.7)
    ax.set_xlabel(r"Test-time DR strength $\lambda$")
    ax.set_ylabel("Crash rate")
    ax.set_title("Det (solid) vs Stochastic (dashed) — T1 OOD")
    ax.legend(loc="upper left", framealpha=0.85, edgecolor="none", fontsize=6, ncol=2)
    ax.set_xlim(-0.05, 1.05)
    plt.tight_layout(pad=0.5)
    fig.savefig(f"{OUT}/c3_crash_det_vs_stoch.pdf")
    fig.savefig(f"{OUT}/c3_crash_det_vs_stoch.png", dpi=160)
    plt.close(fig)
    print(f"[+] wrote {OUT}/c3_crash_det_vs_stoch.pdf")

    # Verdict
    print()
    print("=" * 76)
    print("C3 verdict — does sparse OOD safety survive stochastic evaluation?")
    print("=" * 76)
    for (gd, gs) in [("A", "D"), ("A", "B"), ("A", "E")]:
        det_diffs, sto_diffs = [], []
        for s in SEEDS:
            if (gd, s, 1.0) in det and (gs, s, 1.0) in det:
                det_diffs.append(det[(gd, s, 1.0)] - det[(gs, s, 1.0)])
            if (gd, s, 1.0) in sto and (gs, s, 1.0) in sto:
                sto_diffs.append(sto[(gd, s, 1.0)] - sto[(gs, s, 1.0)])
        dm, dt, _ = paired_t(np.array(det_diffs))
        sm, st, _ = paired_t(np.array(sto_diffs))
        ret = (sm / dm * 100) if dm != 0 else float("nan")
        print(f"  {gd}−{gs} gap @λ=1.0:  det={dm:+.3f} ({sig(dt)})  "
              f"stoch={sm:+.3f} ({sig(st)})  retention={ret:.0f}%")
    da = np.mean([det[("A", s, 1.0)] for s in SEEDS if ("A", s, 1.0) in det])
    sa = np.mean([sto[("A", s, 1.0)] for s in SEEDS if ("A", s, 1.0) in sto])
    dd = np.mean([det[("D", s, 1.0)] for s in SEEDS if ("D", s, 1.0) in det])
    sd = np.mean([sto[("D", s, 1.0)] for s in SEEDS if ("D", s, 1.0) in sto])
    print(f"\n  A crash @λ=1.0: det {da:.3f} -> stoch {sa:.3f}")
    print(f"  D crash @λ=1.0: det {dd:.3f} -> stoch {sd:.3f}")
    print(f"\n  Interpretation: if D-A retention >= ~70% and still significant,")
    print(f"  the C1 safety finding is robust to exploration noise (limitation removed).")
    print(f"  If retention collapses, §VI-D limitation must be tightened honestly.")


if __name__ == "__main__":
    main()
