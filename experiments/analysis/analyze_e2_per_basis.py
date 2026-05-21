"""
E2 — Per-basis leave-one-out necessity analysis.

For each PSC basis i (0=vel_err, 1=omega, 2=tilt, 3=PID_integral, 4=saturation),
group D was retrained with that basis disabled (w_i forced to 0 at every forward),
5 seeds x 1B steps. We compare the resulting success-rate plateau to the full-D
baseline (all 5 bases active), to quantify each basis's individual necessity.

Baseline D (full PSC, sparse, T1): SR plateau from ablation_D_sparse_psc/seed_*/
Ablations: ablation_D_sparse_psc_noPhi{0..4}/seed_*/

Metric: last-10% success_rate plateau (reward-type-independent, same metric as §V-A).
Stats: paired t-test (5 seeds), Cohen's d, relative drop %.

Honest reporting: if a basis removal yields n.s. drop, that basis is empirically
redundant under our training budget — reported as such, not hidden.

Output: experiments/_e2_per_basis/
    e2_table.md
    e2_plateau_bars.{pdf,png}
    e2_sr_curves.{pdf,png}
"""
import glob
import json
import os
from typing import Dict, List, Tuple

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

SEEDS = [42, 123, 456, 789, 1024]
BASIS_NAME = {
    0: "phi0 vel_err",
    1: "phi1 omega",
    2: "phi2 tilt",
    3: "phi3 PID_integral",
    4: "phi4 saturation",
}
OUT_DIR = "experiments/_e2_per_basis"


def stitch(glob_pat: str, key: str = "success_rate") -> Tuple[np.ndarray, np.ndarray]:
    pairs = []
    for p in sorted(glob.glob(glob_pat)):
        try:
            with open(p) as f:
                d = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            continue
        if key not in d:
            continue
        for st, v in zip(d[key]["steps"], d[key]["values"]):
            pairs.append((st, v))
    seen = set()
    out_s, out_v = [], []
    for st, v in sorted(pairs, key=lambda x: x[0]):
        if st not in seen:
            seen.add(st)
            out_s.append(st)
            out_v.append(v)
    return np.array(out_s), np.array(out_v)


def plateau(glob_pat: str, key: str = "success_rate", frac: float = 0.1) -> float:
    _, v = stitch(glob_pat, key)
    if len(v) == 0:
        return float("nan")
    n = max(1, int(len(v) * frac))
    return float(np.mean(v[-n:]))


def paired_t(diffs: np.ndarray) -> Tuple[float, float, float]:
    n = len(diffs)
    if n < 2:
        return float("nan"), float("nan"), float("nan")
    m = float(np.mean(diffs))
    sd = float(np.std(diffs, ddof=1))
    if sd == 0:
        return m, float("inf"), float("inf")
    return m, m / (sd / np.sqrt(n)), m / sd


def sig_marker(t: float) -> str:
    at = abs(t)
    if at > 4.604:
        return "***"   # p<0.01, df=4
    if at > 2.776:
        return "**"    # p<0.05, df=4
    if at > 2.132:
        return "*"     # p<0.10, df=4
    return "n.s."


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    BASE = "experiments/ablation_D_sparse_psc/seed_{s}/*/*/metrics.json"
    ABL = "experiments/ablation_D_sparse_psc_noPhi{i}/seed_{s}/*/*/metrics.json"

    # Per-seed plateaus
    base_plat = {}
    for s in SEEDS:
        base_plat[s] = plateau(BASE.format(s=s))

    abl_plat = {i: {} for i in range(5)}
    for i in range(5):
        for s in SEEDS:
            abl_plat[i][s] = plateau(ABL.format(i=i, s=s))

    base_arr = np.array([base_plat[s] for s in SEEDS])

    lines = []
    lines.append("# E2 — Per-Basis Leave-One-Out Necessity (5 seeds x 1B steps)\n")
    lines.append("Baseline = full PSC group D (all 5 bases). Each row removes one basis.\n")
    lines.append("Metric: last-10% success_rate plateau (same as paper SS V-A).\n")
    lines.append(f"\n**Baseline D (full PSC):** per-seed SR plateau = "
                 f"{', '.join(f'{base_plat[s]:.4f}' for s in SEEDS)}")
    lines.append(f"\n**Baseline D mean +/- SD:** {base_arr.mean():.4f} +/- {base_arr.std(ddof=1):.4f}\n")

    lines.append("\n## Table E2-I. Per-basis ablation vs full-D baseline\n")
    lines.append("| Removed basis | Abl SR (mean +/- SD) | Delta vs D | rel drop % | t (df=4) | Cohen d | verdict |")
    lines.append("|:--|:--:|:--:|:--:|:--:|:--:|:--:|")

    verdicts = {}
    for i in range(5):
        abl_arr = np.array([abl_plat[i][s] for s in SEEDS])
        # paired diff: ablation - baseline (negative = removing the basis HURTS)
        diffs = abl_arr - base_arr
        m, t, d = paired_t(diffs)
        rel = (m / base_arr.mean() * 100.0) if base_arr.mean() != 0 else float("nan")
        sig = sig_marker(t)
        # "necessary" if removing it significantly drops SR
        if sig != "n.s." and m < 0:
            verdict = "NECESSARY"
        elif sig != "n.s." and m > 0:
            verdict = "HARMFUL(!)"
        else:
            verdict = "redundant"
        verdicts[i] = verdict
        lines.append(
            f"| {BASIS_NAME[i]} | {abl_arr.mean():.4f} +/- {abl_arr.std(ddof=1):.4f} "
            f"| {m:+.4f} | {rel:+.1f}% | {t:+.2f} | {d:+.2f} | {sig} {verdict} |"
        )

    # Ranking by harm of removal
    lines.append("\n## Table E2-II. Necessity ranking (most necessary = largest SR drop when removed)\n")
    lines.append("| Rank | Basis | Delta SR when removed | verdict |")
    lines.append("|:--:|:--|:--:|:--:|")
    ranked = sorted(range(5),
                    key=lambda i: np.mean([abl_plat[i][s] for s in SEEDS]) - base_arr.mean())
    for rank, i in enumerate(ranked, 1):
        abl_arr = np.array([abl_plat[i][s] for s in SEEDS])
        m = abl_arr.mean() - base_arr.mean()
        lines.append(f"| {rank} | {BASIS_NAME[i]} | {m:+.4f} | {verdicts[i]} |")

    # Per-seed raw table for transparency
    lines.append("\n## Table E2-III. Raw per-seed SR plateaus\n")
    lines.append("| Config | " + " | ".join(f"seed {s}" for s in SEEDS) + " | mean |")
    lines.append("|:--|" + ":--:|" * (len(SEEDS) + 1))
    lines.append(f"| D (full) | " + " | ".join(f"{base_plat[s]:.4f}" for s in SEEDS)
                 + f" | {base_arr.mean():.4f} |")
    for i in range(5):
        row = " | ".join(f"{abl_plat[i][s]:.4f}" for s in SEEDS)
        mn = np.mean([abl_plat[i][s] for s in SEEDS])
        lines.append(f"| noPhi{i} ({BASIS_NAME[i]}) | {row} | {mn:.4f} |")

    with open(f"{OUT_DIR}/e2_table.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[+] wrote {OUT_DIR}/e2_table.md")

    # ---- Figures ----
    # Bar chart: D baseline + 5 ablations
    fig, ax = plt.subplots(1, 1, figsize=(COL_WIDTH, ROW_H))
    labels = ["D\n(full)"] + [f"-phi{i}" for i in range(5)]
    means = [base_arr.mean()] + [np.mean([abl_plat[i][s] for s in SEEDS]) for i in range(5)]
    sds = [base_arr.std(ddof=1)] + [np.std([abl_plat[i][s] for s in SEEDS], ddof=1) for i in range(5)]
    colors = ["#22aa44"] + ["#cc6644"] * 5
    ax.bar(range(6), means, yerr=sds, color=colors, edgecolor="black", linewidth=0.4, capsize=2)
    ax.axhline(base_arr.mean(), color="#22aa44", lw=0.8, ls="--", alpha=0.6)
    ax.set_xticks(range(6))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Success-rate plateau")
    ax.set_title("Per-basis leave-one-out (T1, 5 seeds x 1B)")
    plt.tight_layout(pad=0.5)
    fig.savefig(f"{OUT_DIR}/e2_plateau_bars.pdf")
    fig.savefig(f"{OUT_DIR}/e2_plateau_bars.png", dpi=160)
    plt.close(fig)
    print(f"[+] wrote {OUT_DIR}/e2_plateau_bars.pdf")

    # SR convergence curves
    fig, ax = plt.subplots(1, 1, figsize=(DBL_WIDTH * 0.6, ROW_H))
    cmap = ["#cc4444", "#dd8800", "#3377cc", "#aa44aa", "#888811"]
    # baseline curve
    per_seed = []
    common = None
    for s in SEEDS:
        st, v = stitch(BASE.format(s=s))
        if len(v) == 0:
            continue
        per_seed.append((st, v))
        if common is None or len(st) > len(common):
            common = st
    if per_seed and common is not None:
        arr = np.array([np.interp(common, st, v, left=v[0], right=v[-1]) for st, v in per_seed])
        ax.plot(common / 1e9, np.median(arr, axis=0), color="#22aa44", lw=1.6, label="D (full)")
    for i in range(5):
        per_seed = []
        common = None
        for s in SEEDS:
            st, v = stitch(ABL.format(i=i, s=s))
            if len(v) == 0:
                continue
            per_seed.append((st, v))
            if common is None or len(st) > len(common):
                common = st
        if not per_seed or common is None:
            continue
        arr = np.array([np.interp(common, st, v, left=v[0], right=v[-1]) for st, v in per_seed])
        ax.plot(common / 1e9, np.median(arr, axis=0), color=cmap[i], lw=1.1,
                label=f"-phi{i} {BASIS_NAME[i].split()[1]}")
    ax.set_xlabel("Env steps (1e9)")
    ax.set_ylabel("Success rate")
    ax.set_xlim(0, 1.0)
    ax.legend(loc="best", framealpha=0.85, edgecolor="none", fontsize=7)
    ax.set_title("E2 per-basis SR convergence (5-seed median)")
    plt.tight_layout(pad=0.5)
    fig.savefig(f"{OUT_DIR}/e2_sr_curves.pdf")
    fig.savefig(f"{OUT_DIR}/e2_sr_curves.png", dpi=160)
    plt.close(fig)
    print(f"[+] wrote {OUT_DIR}/e2_sr_curves.pdf")

    # ---- Verdict to stdout ----
    print()
    print("=" * 76)
    print("E2 verdict — per-basis necessity (paired vs full-D, 5 seeds)")
    print("=" * 76)
    print(f"  Baseline D (full PSC): SR = {base_arr.mean():.4f} +/- {base_arr.std(ddof=1):.4f}")
    print()
    for i in range(5):
        abl_arr = np.array([abl_plat[i][s] for s in SEEDS])
        diffs = abl_arr - base_arr
        m, t, d = paired_t(diffs)
        rel = m / base_arr.mean() * 100.0
        print(f"  remove {BASIS_NAME[i]:18s}: SR={abl_arr.mean():.4f}  "
              f"d_SR={m:+.4f} ({rel:+.1f}%)  t={t:+.2f}  d={d:+.2f}  "
              f"{sig_marker(t)} -> {verdicts[i]}")
    print()
    n_nec = sum(1 for v in verdicts.values() if v == "NECESSARY")
    print(f"  Summary: {n_nec}/5 bases empirically necessary "
          f"(significant SR drop when removed at 5-seed x 1B rigor).")


if __name__ == "__main__":
    main()
