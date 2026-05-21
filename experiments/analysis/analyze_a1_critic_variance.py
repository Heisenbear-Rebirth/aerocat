"""
A1 — Critic TD Variance Reduction Analysis (v19.4 follow-up)

Tests v19.3's claim that PSC reduces critic-side variance via control-variate decomposition.

Public ground-truth metric: `td_error_std` = std of standard TD error `r + γV(s') - V(s)`.
This is logged identically for MLP and PSC groups (no semantic skew), so within-reward
contrasts (A vs C vs F, dense; B vs D, sparse) are apples-to-apples.

`td_phys_std` is NOT used for the cross-group comparison because for MLP groups it
degenerates to `std(reward + 0 - 0) ≈ std(reward)` (PSC layer outputs 0), while for
PSC groups it is the V_phys Bellman residual. Not comparable across MLP/PSC.

`advantage_std` is the GAE output std after value bootstrap, also comparable.

`v_phys_ratio` is the |V_phys|/|V_total| mean — only meaningful for PSC groups (C/D/E);
plotted to show how much of V the structural prior carries during training.

Outputs:
  experiments/_a1_variance/
    a1_table.md                — group means/sds + within-reward paired contrasts
    a1_td_error_std_curves.{pdf,png}      — 3-panel (T1/T2/T3), td_error_std vs steps
    a1_advantage_std_curves.{pdf,png}     — 3-panel, advantage_std vs steps
    a1_vphys_ratio_curves.{pdf,png}       — 1-panel for T1, only PSC groups
    a1_plateau_bars.{pdf,png}             — last-10% plateau bars per group/task

Usage (from v18/):
    python experiments/_scripts/analyze_a1_critic_variance.py
"""
import glob
import json
import os
import statistics
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

# ============================================================
# IEEE styling (matches make_paper_figures.py)
# ============================================================
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

COLORS = {
    "A": "#666666",
    "B": "#cc4444",
    "C": "#3377cc",
    "D": "#22aa44",
    "E": "#aa44aa",
    "F": "#dd8800",
}
LABELS = {
    "A": "A: MLP+dense",
    "B": "B: MLP+sparse",
    "C": "C: PSC+dense",
    "D": "D: PSC+sparse",
    "E": "E: PSC$_{\\mathrm{fix-}w}$+sparse",
    "F": "F: Cai 2025 dual",
}
GROUP_PATH = {
    "A": ("dense", "mlp"),
    "B": ("sparse", "mlp"),
    "C": ("dense", "psc"),
    "D": ("sparse", "psc"),
    "E": ("sparse", "psc_fixedw"),
    "F": ("dense", "mlp_dual"),
}
SEEDS = [42, 123, 456, 789, 1024]

# Group coverage per task
TASK_GROUPS = {
    "velocity":    ["A", "B", "C", "D", "E", "F"],
    "waypoint":    ["A", "B", "C", "D", "F"],
    "disturbance": ["A", "C", "F"],
}
TASK_LABEL = {"velocity": "T1 (velocity)", "waypoint": "T2 (waypoint)", "disturbance": "T3 (disturbance)"}


def task_suffix(task: str) -> str:
    return "" if task == "velocity" else f"_{task}"


def all_metrics_paths(g: str, s: int, task: str) -> List[str]:
    rt, ct = GROUP_PATH[g]
    suf = task_suffix(task)
    return sorted(glob.glob(f"experiments/ablation_{g}_{rt}_{ct}{suf}/seed_{s}/*/*/metrics.json"))


def stitch(g: str, s: int, key: str, task: str) -> Tuple[np.ndarray, np.ndarray]:
    pairs = []
    for p in all_metrics_paths(g, s, task):
        try:
            with open(p) as f:
                d = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            continue
        if key not in d:
            continue
        e = d[key]
        for st, v in zip(e.get("steps", []), e.get("values", [])):
            pairs.append((st, v))
    seen = set()
    out_s, out_v = [], []
    for st, v in sorted(pairs, key=lambda x: x[0]):
        if st not in seen:
            seen.add(st)
            out_s.append(st)
            out_v.append(v)
    return np.array(out_s), np.array(out_v)


def aggregate(g: str, task: str, key: str) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """Return (steps, [seed, step] array, list of seeds with data)."""
    per_seed = []
    used_seeds = []
    common_steps = None
    for s in SEEDS:
        steps, vals = stitch(g, s, key, task)
        if len(vals) == 0:
            continue
        per_seed.append((steps, vals))
        used_seeds.append(s)
        if common_steps is None or len(steps) > len(common_steps):
            common_steps = steps
    if not per_seed:
        return np.array([]), np.array([[]]), []
    interp = np.array([
        np.interp(common_steps, st, v, left=v[0], right=v[-1])
        for st, v in per_seed
    ])
    return common_steps, interp, used_seeds


def plateau(g: str, s: int, task: str, key: str, frac: float = 0.1) -> float:
    _, vals = stitch(g, s, key, task)
    if len(vals) == 0:
        return float("nan")
    n = max(1, int(len(vals) * frac))
    return float(np.mean(vals[-n:]))


def paired_t(diffs: np.ndarray) -> Tuple[float, float, float]:
    """Returns (mean_diff, t, cohen_d). diffs is per-seed array."""
    n = len(diffs)
    if n < 2:
        return float("nan"), float("nan"), float("nan")
    m = float(np.mean(diffs))
    sd = float(np.std(diffs, ddof=1))
    if sd == 0:
        return m, float("inf"), float("inf")
    t = m / (sd / np.sqrt(n))
    d = m / sd
    return m, t, d


# ============================================================
# Main analysis
# ============================================================
def main():
    out_dir = "experiments/_a1_variance"
    os.makedirs(out_dir, exist_ok=True)

    METRICS = ["td_error_std", "advantage_std", "vf_loss", "v_phys_ratio", "td_phys_std", "td_res_std"]

    # ----- Collect plateau values per (group, task, seed, metric) -----
    plats: Dict[str, Dict[str, Dict[str, Dict[int, float]]]] = {}
    for task, groups in TASK_GROUPS.items():
        for g in groups:
            for s in SEEDS:
                for k in METRICS:
                    plats.setdefault(k, {}).setdefault(task, {}).setdefault(g, {})[s] = plateau(g, s, task, k)

    # ----- Write table -----
    lines = []
    lines.append("# A1 — Critic TD Variance Analysis (last-10% plateau, 5-seed mean ± SD)\n")
    lines.append("Generated by `analyze_a1_critic_variance.py`. Within-reward contrasts only.\n")

    for k in ["td_error_std", "advantage_std", "vf_loss"]:
        lines.append(f"\n## {k} (last-10% plateau, 5-seed mean ± SD)\n")
        lines.append("| Group |  T1 (velocity)   |  T2 (waypoint)   |  T3 (disturbance) |")
        lines.append("|:---:|:---:|:---:|:---:|")
        for g in "ABCDEF":
            cells = []
            for task in ["velocity", "waypoint", "disturbance"]:
                if task not in plats[k] or g not in plats[k][task]:
                    cells.append("—")
                    continue
                vals = np.array([plats[k][task][g][s] for s in SEEDS if s in plats[k][task][g] and not np.isnan(plats[k][task][g][s])])
                if len(vals) == 0:
                    cells.append("—")
                else:
                    cells.append(f"{vals.mean():.3g} ± {vals.std(ddof=1):.2g}")
            lines.append(f"| {g} | {cells[0]} | {cells[1]} | {cells[2]} |")

    # Within-reward contrasts for td_error_std
    lines.append("\n## Within-reward paired contrasts (td_error_std plateau)\n")
    lines.append("Positive Δ ⇒ PSC/variant raises TD-error std (worse, counter to v19.3 claim).")
    lines.append("Negative Δ ⇒ PSC/variant lowers TD-error std (supports control-variate claim).\n")
    lines.append("| Task | Contrast | Δ td_error_std | t (df=4) | Cohen d | Direction |")
    lines.append("|:---:|:---:|:---:|:---:|:---:|:---:|")

    contrasts = [
        ("velocity", "C - A", "C", "A", "dense"),
        ("velocity", "F - A", "F", "A", "dense"),
        ("velocity", "C - F", "C", "F", "dense"),
        ("velocity", "D - B", "D", "B", "sparse"),
        ("velocity", "E - B", "E", "B", "sparse"),
        ("velocity", "D - E", "D", "E", "sparse"),
        ("waypoint", "C - A", "C", "A", "dense"),
        ("waypoint", "F - A", "F", "A", "dense"),
        ("waypoint", "C - F", "C", "F", "dense"),
        ("waypoint", "D - B", "D", "B", "sparse"),
        ("disturbance", "C - A", "C", "A", "dense"),
        ("disturbance", "F - A", "F", "A", "dense"),
        ("disturbance", "C - F", "C", "F", "dense"),
    ]
    k = "td_error_std"
    for task, label, ga, gb, _ in contrasts:
        if task not in plats[k] or ga not in plats[k][task] or gb not in plats[k][task]:
            continue
        diffs = []
        for s in SEEDS:
            va = plats[k][task][ga].get(s, float("nan"))
            vb = plats[k][task][gb].get(s, float("nan"))
            if np.isnan(va) or np.isnan(vb):
                continue
            diffs.append(va - vb)
        diffs = np.array(diffs)
        m, t, d = paired_t(diffs)
        if np.isnan(m):
            continue
        sig = "↓ (good)" if m < 0 and abs(t) > 2.776 else ("↑ (bad)" if m > 0 and abs(t) > 2.776 else "n.s.")
        lines.append(f"| {task[:3].upper()} | {label} | {m:+.3g} | {t:+.2f} | {d:+.2f} | {sig} |")

    # v_phys_ratio summary (only PSC groups)
    lines.append("\n## V_phys / V_total ratio (last-10% plateau, PSC groups only)\n")
    lines.append("Shows how much of V is carried by the structural prior at convergence.\n")
    lines.append("| Group | T1 | T2 | T3 |")
    lines.append("|:---:|:---:|:---:|:---:|")
    k = "v_phys_ratio"
    for g in "CDE":
        cells = []
        for task in ["velocity", "waypoint", "disturbance"]:
            if task not in plats[k] or g not in plats[k][task]:
                cells.append("—")
                continue
            vals = np.array([plats[k][task][g][s] for s in SEEDS if s in plats[k][task][g] and not np.isnan(plats[k][task][g][s])])
            if len(vals) == 0 or vals.mean() < 1e-6:
                cells.append("—")
            else:
                cells.append(f"{vals.mean():.3f} ± {vals.std(ddof=1):.3f}")
        lines.append(f"| {g} | {cells[0]} | {cells[1]} | {cells[2]} |")

    with open(f"{out_dir}/a1_table.md", "w") as f:
        f.write("\n".join(lines))
    print(f"[+] wrote {out_dir}/a1_table.md")

    # ----- Figures -----
    # Fig 1: td_error_std time series, 3 panels for tasks
    fig, axes = plt.subplots(1, 3, figsize=(DBL_WIDTH, ROW_H))
    for ax, task in zip(axes, ["velocity", "waypoint", "disturbance"]):
        for g in TASK_GROUPS[task]:
            steps, arr, _ = aggregate(g, task, "td_error_std")
            if arr.size == 0:
                continue
            med = np.median(arr, axis=0)
            q1 = np.percentile(arr, 25, axis=0)
            q3 = np.percentile(arr, 75, axis=0)
            ax.plot(steps / 1e9, med, label=LABELS[g], color=COLORS[g], lw=1.3)
            ax.fill_between(steps / 1e9, q1, q3, color=COLORS[g], alpha=0.15, lw=0)
        ax.set_title(TASK_LABEL[task])
        ax.set_xlabel("Env steps (1e9)")
        if task == "velocity":
            ax.set_ylabel("td_error_std")
            ax.legend(loc="upper left", framealpha=0.85, edgecolor='none', fontsize=7)
        ax.set_xlim(0, 1.0)
    plt.tight_layout(pad=0.5)
    fig.savefig(f"{out_dir}/a1_td_error_std_curves.pdf"); fig.savefig(f"{out_dir}/a1_td_error_std_curves.png", dpi=160)
    plt.close(fig)
    print(f"[+] wrote {out_dir}/a1_td_error_std_curves.pdf")

    # Fig 2: advantage_std time series
    fig, axes = plt.subplots(1, 3, figsize=(DBL_WIDTH, ROW_H))
    for ax, task in zip(axes, ["velocity", "waypoint", "disturbance"]):
        for g in TASK_GROUPS[task]:
            steps, arr, _ = aggregate(g, task, "advantage_std")
            if arr.size == 0:
                continue
            med = np.median(arr, axis=0)
            q1 = np.percentile(arr, 25, axis=0)
            q3 = np.percentile(arr, 75, axis=0)
            ax.plot(steps / 1e9, med, label=LABELS[g], color=COLORS[g], lw=1.3)
            ax.fill_between(steps / 1e9, q1, q3, color=COLORS[g], alpha=0.15, lw=0)
        ax.set_title(TASK_LABEL[task])
        ax.set_xlabel("Env steps (1e9)")
        if task == "velocity":
            ax.set_ylabel("advantage_std (GAE)")
            ax.legend(loc="upper left", framealpha=0.85, edgecolor='none', fontsize=7)
        ax.set_xlim(0, 1.0)
    plt.tight_layout(pad=0.5)
    fig.savefig(f"{out_dir}/a1_advantage_std_curves.pdf"); fig.savefig(f"{out_dir}/a1_advantage_std_curves.png", dpi=160)
    plt.close(fig)
    print(f"[+] wrote {out_dir}/a1_advantage_std_curves.pdf")

    # Fig 3: v_phys_ratio for PSC groups (T1 only — T2 sparse collapsed, T3 has no D/E)
    fig, ax = plt.subplots(1, 1, figsize=(COL_WIDTH, ROW_H))
    for g in ["C", "D", "E"]:
        steps, arr, _ = aggregate(g, "velocity", "v_phys_ratio")
        if arr.size == 0:
            continue
        med = np.median(arr, axis=0)
        q1 = np.percentile(arr, 25, axis=0)
        q3 = np.percentile(arr, 75, axis=0)
        ax.plot(steps / 1e9, med, label=LABELS[g], color=COLORS[g], lw=1.5)
        ax.fill_between(steps / 1e9, q1, q3, color=COLORS[g], alpha=0.18, lw=0)
    ax.set_title("$|V_{\\mathrm{phys}}|/|V_{\\mathrm{total}}|$ (T1)")
    ax.set_xlabel("Env steps (1e9)")
    ax.set_ylabel("Ratio")
    ax.set_xlim(0, 1.0)
    ax.legend(loc="best", framealpha=0.85, edgecolor='none', fontsize=8)
    plt.tight_layout(pad=0.5)
    fig.savefig(f"{out_dir}/a1_vphys_ratio_curves.pdf"); fig.savefig(f"{out_dir}/a1_vphys_ratio_curves.png", dpi=160)
    plt.close(fig)
    print(f"[+] wrote {out_dir}/a1_vphys_ratio_curves.pdf")

    # Fig 4: plateau bars (td_error_std + advantage_std), 2 panels
    fig, axes = plt.subplots(1, 2, figsize=(DBL_WIDTH, ROW_H * 1.1))
    for ax, k, title in [
        (axes[0], "td_error_std", "td_error_std plateau"),
        (axes[1], "advantage_std", "advantage_std (GAE) plateau"),
    ]:
        x_labels = []
        means = []
        sds = []
        colors = []
        gap = 0
        x_pos = []
        cur = 0
        for task in ["velocity", "waypoint", "disturbance"]:
            for g in TASK_GROUPS[task]:
                if task not in plats[k] or g not in plats[k][task]:
                    continue
                vals = np.array([plats[k][task][g][s] for s in SEEDS if s in plats[k][task][g] and not np.isnan(plats[k][task][g][s])])
                if len(vals) == 0:
                    continue
                x_pos.append(cur)
                cur += 1
                means.append(vals.mean())
                sds.append(vals.std(ddof=1))
                colors.append(COLORS[g])
                x_labels.append(f"{g}\n{task[:3].upper()}")
            cur += 0.8  # gap between tasks
        ax.bar(x_pos, means, yerr=sds, color=colors, edgecolor='black', linewidth=0.4, capsize=2)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels, fontsize=7)
        ax.set_title(title)
        ax.set_ylabel(k)
    plt.tight_layout(pad=0.5)
    fig.savefig(f"{out_dir}/a1_plateau_bars.pdf"); fig.savefig(f"{out_dir}/a1_plateau_bars.png", dpi=160)
    plt.close(fig)
    print(f"[+] wrote {out_dir}/a1_plateau_bars.pdf")

    # ----- Print verdict to stdout -----
    print()
    print("=" * 72)
    print("A1 verdict — within-reward td_error_std contrasts at plateau")
    print("=" * 72)
    for task, label, ga, gb, _ in contrasts:
        k = "td_error_std"
        if task not in plats[k] or ga not in plats[k][task] or gb not in plats[k][task]:
            continue
        diffs = []
        for s in SEEDS:
            va = plats[k][task][ga].get(s, float("nan"))
            vb = plats[k][task][gb].get(s, float("nan"))
            if not (np.isnan(va) or np.isnan(vb)):
                diffs.append(va - vb)
        diffs = np.array(diffs)
        m, t, d = paired_t(diffs)
        if np.isnan(m):
            continue
        verdict = ""
        if abs(t) > 2.776:
            verdict = "  ↓ PSC REDUCES variance" if m < 0 else "  ↑ PSC INCREASES variance"
        else:
            verdict = "  n.s."
        print(f"  {task[:3].upper()} {label:<6}  Δ={m:+8.3f}  t={t:+6.2f}  d={d:+6.2f}{verdict}")


if __name__ == "__main__":
    main()
