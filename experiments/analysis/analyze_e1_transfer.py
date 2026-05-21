"""
E1 — Cross-task PSC weight transfer analysis.

Two configs on T3 (disturbance), group D (PSC + sparse):
  baseline : ablation_D_sparse_psc_disturbance/seed_*/...   (default init [45,2,2,0.5,1]/20)
  transfer : ablation_D_sparse_psc_disturbance_initT1transfer/seed_*/...
             (init = D-T1 converged 5-seed mean: [44.10, 2.32, -0.02, -0.54, 0.06]/22.68)

Three questions:
  1. Does T3-sparse training converge at all? (P1 gap)
       Compare SR plateau to T3-dense (A/C/F ~ 0.534) and T1-sparse (D ~ 0.335).
       If T3-sparse plateau >> 0.02 (the T2-sparse collapse number) -> sparse
       precondition holds on T3.
  2. Does T1-transferred init give cold-start speedup over default init?
       Steps-to-SR-threshold (e.g. 0.1, 0.2, 0.3) — paired t-test per seed.
  3. Does the final plateau differ between baseline and transfer? Paired t.

Outputs: experiments/_e1_transfer/
    e1_table.md
    e1_sr_curves.{pdf,png}     -- T3 baseline vs transfer (5-seed median + IQR), plus T1 D reference
    e1_plateau_bars.{pdf,png}  -- T3 baseline / transfer / T1 D / T3 dense (A) plateaus
"""
import glob, json, os
from typing import List, Tuple, Dict

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
OUT = "experiments/_e1_transfer"

# Glob patterns: {s} placeholder for seed
DIRS = {
    "baseline_T3":  "experiments/ablation_D_sparse_psc_disturbance/seed_{s}/*/*/metrics.json",
    "transfer_T3":  "experiments/ablation_D_sparse_psc_disturbance_initT1transfer/seed_{s}/*/*/metrics.json",
    "D_T1_ref":     "experiments/ablation_D_sparse_psc/seed_{s}/*/*/metrics.json",
    "A_T3_ref":     "experiments/ablation_A_dense_mlp_disturbance/seed_{s}/*/*/metrics.json",
}


def stitch(pat, key="success_rate"):
    pairs = []
    for p in sorted(glob.glob(pat)):
        try:
            d = json.load(open(p))
        except (FileNotFoundError, json.JSONDecodeError):
            continue
        if key not in d:
            continue
        for st, v in zip(d[key]["steps"], d[key]["values"]):
            pairs.append((st, v))
    if not pairs:
        return np.array([]), np.array([])
    pairs.sort(key=lambda x: x[0])
    seen, out_s, out_v = set(), [], []
    for st, v in pairs:
        if st not in seen:
            seen.add(st); out_s.append(st); out_v.append(v)
    return np.array(out_s), np.array(out_v)


def plateau(pat, key="success_rate", frac=0.1):
    _, v = stitch(pat, key)
    if len(v) == 0:
        return float("nan")
    n = max(1, int(len(v) * frac))
    return float(np.mean(v[-n:]))


def time_to_threshold(pat, threshold, key="success_rate"):
    """First env-step where success_rate >= threshold (linear interp)."""
    steps, vals = stitch(pat, key)
    if len(vals) == 0:
        return float("nan")
    above = vals >= threshold
    if not above.any():
        return float("nan")
    idx = int(np.argmax(above))
    if idx == 0:
        return float(steps[0])
    v0, v1 = vals[idx - 1], vals[idx]
    s0, s1 = steps[idx - 1], steps[idx]
    if v1 == v0:
        return float(s1)
    frac = (threshold - v0) / (v1 - v0)
    return float(s0 + frac * (s1 - s0))


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

    # ---- Plateaus per (config, seed) ----
    plats = {}
    for cfg, pat in DIRS.items():
        plats[cfg] = {}
        for s in SEEDS:
            plats[cfg][s] = plateau(pat.format(s=s))

    lines = []
    lines.append("# E1 — Cross-Task PSC Weight Transfer Analysis\n")
    lines.append("Two T3-sparse configs: baseline (default PSC init) vs transfer (D-T1 converged init).\n")
    lines.append("Reference: D-T1 (T1-sparse, full PSC) and A-T3 (T3-dense, MLP baseline).\n")

    lines.append("\n## Table E1-I. Per-seed SR plateau\n")
    lines.append("| Config | " + " | ".join(f"seed {s}" for s in SEEDS) + " | mean ± SD |")
    lines.append("|:--|" + ":--:|" * (len(SEEDS) + 1))
    for cfg in ["baseline_T3", "transfer_T3", "D_T1_ref", "A_T3_ref"]:
        vs = [plats[cfg][s] for s in SEEDS]
        valid = [v for v in vs if not np.isnan(v)]
        if not valid:
            lines.append(f"| {cfg} | " + " | ".join("—" for _ in SEEDS) + " | — |")
            continue
        m = np.mean(valid); sd = np.std(valid, ddof=1) if len(valid) > 1 else 0.0
        row = " | ".join(f"{v:.4f}" if not np.isnan(v) else "—" for v in vs)
        lines.append(f"| {cfg} | {row} | {m:.4f} ± {sd:.4f} |")

    # ---- Q1: T3-sparse precondition check ----
    lines.append("\n## Q1: T3-sparse precondition (does sparse converge on T3?)\n")
    base_sr = np.array([plats["baseline_T3"][s] for s in SEEDS if not np.isnan(plats["baseline_T3"][s])])
    d_t1_sr = np.array([plats["D_T1_ref"][s] for s in SEEDS if not np.isnan(plats["D_T1_ref"][s])])
    a_t3_sr = np.array([plats["A_T3_ref"][s] for s in SEEDS if not np.isnan(plats["A_T3_ref"][s])])
    lines.append(f"- **T3-sparse (baseline) SR plateau** = {base_sr.mean():.4f} ± {base_sr.std(ddof=1):.4f}")
    lines.append(f"- T1-sparse (D, full PSC) SR plateau = {d_t1_sr.mean():.4f} ± {d_t1_sr.std(ddof=1):.4f}")
    lines.append(f"- T3-dense (A) SR plateau = {a_t3_sr.mean():.4f} ± {a_t3_sr.std(ddof=1):.4f}")
    lines.append(f"- T2-sparse (D, from v19.4 deliverable) = 0.026 ± 0.014 (collapse reference)")
    if base_sr.mean() > 0.05:
        verdict = (f"**T3-sparse converges to non-degenerate plateau** ({base_sr.mean():.3f} >> 0.026 T2 collapse). "
                   f"Sparse precondition holds on T3.")
    else:
        verdict = (f"**T3-sparse collapses** ({base_sr.mean():.3f} ~ 0.026 T2 collapse). "
                   f"Sparse precondition fails on T3 — applicability boundary extends.")
    lines.append(f"\n**Verdict**: {verdict}")

    # ---- Q2: Cold-start speedup transfer vs baseline ----
    lines.append("\n## Q2: Cold-start speedup — transfer vs baseline (paired seeds)\n")
    lines.append("| Threshold SR | baseline median steps | transfer median steps | speedup b/t |")
    lines.append("|:--:|:--:|:--:|:--:|")
    speedups = {}
    for th in [0.05, 0.1, 0.2, 0.3]:
        bs = []
        ts = []
        for s in SEEDS:
            b = time_to_threshold(DIRS["baseline_T3"].format(s=s), th)
            t = time_to_threshold(DIRS["transfer_T3"].format(s=s), th)
            if not np.isnan(b):
                bs.append(b)
            if not np.isnan(t):
                ts.append(t)
        if not bs or not ts:
            lines.append(f"| {th:.2f} | — | — | — |")
            continue
        mb = np.median(bs)
        mt = np.median(ts)
        speed = mb / mt if mt > 0 else float("nan")
        speedups[th] = speed
        lines.append(f"| {th:.2f} | {mb/1e6:.0f}M ({len(bs)}/5) | {mt/1e6:.0f}M ({len(ts)}/5) | {speed:.2f}× |")

    # Paired t on time-to-SR=0.1 (cold-start exit)
    lines.append("\n### Paired t-test on time-to-SR-threshold (transfer − baseline; negative = transfer faster)\n")
    lines.append("| Threshold | Δ (transfer − baseline, M steps) | t (df=4) | Cohen d | sig |")
    lines.append("|:--:|:--:|:--:|:--:|:--:|")
    for th in [0.05, 0.1, 0.2, 0.3]:
        diffs = []
        for s in SEEDS:
            b = time_to_threshold(DIRS["baseline_T3"].format(s=s), th)
            t = time_to_threshold(DIRS["transfer_T3"].format(s=s), th)
            if not (np.isnan(b) or np.isnan(t)):
                diffs.append((t - b) / 1e6)  # in M steps
        if len(diffs) < 2:
            continue
        m, t_stat, d = paired_t(np.array(diffs))
        lines.append(f"| {th:.2f} | {m:+.1f} | {t_stat:+.2f} | {d:+.2f} | {sig(t_stat)} |")

    # ---- Q3: Plateau difference transfer vs baseline ----
    lines.append("\n## Q3: Final plateau — transfer vs baseline (paired seeds)\n")
    diffs = []
    for s in SEEDS:
        b = plats["baseline_T3"][s]; t = plats["transfer_T3"][s]
        if not (np.isnan(b) or np.isnan(t)):
            diffs.append(t - b)
    if len(diffs) >= 2:
        m, t_stat, d = paired_t(np.array(diffs))
        lines.append(f"- Δ SR plateau (transfer − baseline) = {m:+.4f}  t={t_stat:+.2f}  d={d:+.2f}  {sig(t_stat)}")
        lines.append(f"- Per-seed Δ: {['%+.4f' % x for x in diffs]}")
    else:
        lines.append("- insufficient data")

    with open(f"{OUT}/e1_table.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[+] wrote {OUT}/e1_table.md")

    # ---- Figures ----
    # SR convergence curves: baseline T3, transfer T3, D T1 ref
    fig, ax = plt.subplots(1, 1, figsize=(COL_WIDTH * 1.3, ROW_H))
    for cfg, color, label in [
        ("baseline_T3", "#cc6644", "D T3-sparse (default init)"),
        ("transfer_T3", "#22aa44", "D T3-sparse (T1-transfer init)"),
        ("D_T1_ref", "#3377cc", "D T1-sparse (reference)"),
    ]:
        per_seed = []
        common = None
        for s in SEEDS:
            steps, vals = stitch(DIRS[cfg].format(s=s))
            if len(vals) == 0:
                continue
            per_seed.append((steps, vals))
            if common is None or len(steps) > len(common):
                common = steps
        if not per_seed or common is None:
            continue
        arr = np.array([np.interp(common, st, v, left=v[0], right=v[-1]) for st, v in per_seed])
        med = np.median(arr, axis=0)
        q1 = np.percentile(arr, 25, axis=0)
        q3 = np.percentile(arr, 75, axis=0)
        ax.plot(common / 1e9, med, color=color, lw=1.4, label=label)
        ax.fill_between(common / 1e9, q1, q3, color=color, alpha=0.18, lw=0)
    ax.set_xlabel("Env steps (1e9)")
    ax.set_ylabel("Success rate")
    ax.set_xlim(0, 1.0)
    ax.legend(loc="best", framealpha=0.85, edgecolor="none", fontsize=7)
    ax.set_title("E1 cross-task PSC weight transfer (T3-sparse)")
    plt.tight_layout(pad=0.5)
    fig.savefig(f"{OUT}/e1_sr_curves.pdf"); fig.savefig(f"{OUT}/e1_sr_curves.png", dpi=160)
    plt.close(fig)
    print(f"[+] wrote {OUT}/e1_sr_curves.pdf")

    # ---- Verdict ----
    print()
    print("=" * 78)
    print("E1 verdict")
    print("=" * 78)
    print(f"  baseline T3-sparse SR = {base_sr.mean():.4f} ± {base_sr.std(ddof=1):.4f}")
    print(f"  transfer T3-sparse SR = {np.mean([plats['transfer_T3'][s] for s in SEEDS]):.4f}")
    print(f"  T1-sparse (D ref)    = {d_t1_sr.mean():.4f}")
    print(f"  T2-sparse collapse   = 0.026 (deliverable)")
    if len(diffs) >= 2:
        m_p, t_p, d_p = paired_t(np.array(diffs))
        print(f"  transfer − baseline plateau Δ = {m_p:+.4f} ({sig(t_p)})")


if __name__ == "__main__":
    main()
