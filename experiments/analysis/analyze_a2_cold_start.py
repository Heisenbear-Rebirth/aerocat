"""
A2 — Critic Cold-Start Anchoring Analysis (v19.4 follow-up)

After A1 disproved the "PSC reduces TD variance via control variate" claim, this script
tests the replacement hypothesis: PSC anchors the critic during the cold-start phase
(early training, before V_res has had time to absorb structure).

Three indicators:
  (1) Early vf_loss decay: log-ratio log(vf_loss[T] / vf_loss[5M]) at T = 50M, 100M, 200M.
      A more-negative number = critic learning faster. Within-reward only (sparse vs sparse,
      dense vs dense), because vf_loss is reward-scale-specific.
  (2) Early v_phys_ratio average: |V_phys|/|V_total| mean over 5-50M (PSC groups only).
      Tests whether PSC's structural prior carries the value function during cold-start.
  (3) Time-to-SR-threshold: env-steps to first reach SR = 0.1, 0.2, 0.3. Reward-type-
      independent → cross-reward and within-reward both valid.

Within-reward contrasts at plateau (last 10%) are already done in A1; A2 focuses on the
early phase (first 200M of 1B steps = first 20% of training).

Outputs:
  experiments/_a2_cold_start/
    a2_table.md
    a2_vfloss_early.{pdf,png}     — 2-panel: dense (A/C/F) and sparse (B/D/E), vf_loss 0-200M
    a2_vphys_ratio_early.{pdf,png}— PSC groups (C/D/E) v_phys_ratio 0-200M
    a2_sr_threshold.{pdf,png}     — bars: steps-to-SR-threshold per group
"""
import glob
import json
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

# Styling (same as A1)
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
COLORS = {"A": "#666666", "B": "#cc4444", "C": "#3377cc", "D": "#22aa44", "E": "#aa44aa", "F": "#dd8800"}
LABELS = {"A": "A: MLP+dense", "B": "B: MLP+sparse", "C": "C: PSC+dense", "D": "D: PSC+sparse",
          "E": "E: PSC$_{\\mathrm{fix-}w}$+sparse", "F": "F: Cai 2025 dual"}
GROUP_PATH = {"A": ("dense", "mlp"), "B": ("sparse", "mlp"), "C": ("dense", "psc"),
              "D": ("sparse", "psc"), "E": ("sparse", "psc_fixedw"), "F": ("dense", "mlp_dual")}
SEEDS = [42, 123, 456, 789, 1024]
TASK_GROUPS = {
    "velocity":    ["A", "B", "C", "D", "E", "F"],
    "waypoint":    ["A", "B", "C", "D", "F"],
    "disturbance": ["A", "C", "F"],
}
TASK_LABEL = {"velocity": "T1", "waypoint": "T2", "disturbance": "T3"}


def task_suffix(task):
    return "" if task == "velocity" else f"_{task}"


def all_paths(g, s, task):
    rt, ct = GROUP_PATH[g]
    suf = task_suffix(task)
    return sorted(glob.glob(f"experiments/ablation_{g}_{rt}_{ct}{suf}/seed_{s}/*/*/metrics.json"))


def stitch(g, s, key, task):
    pairs = []
    for p in all_paths(g, s, task):
        try:
            with open(p) as f: d = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            continue
        if key not in d: continue
        for st, v in zip(d[key]["steps"], d[key]["values"]):
            pairs.append((st, v))
    seen = set(); out_s = []; out_v = []
    for st, v in sorted(pairs, key=lambda x: x[0]):
        if st not in seen:
            seen.add(st); out_s.append(st); out_v.append(v)
    return np.array(out_s), np.array(out_v)


def val_at(g, s, key, task, target_step):
    steps, vals = stitch(g, s, key, task)
    if len(vals) == 0: return float("nan")
    return float(np.interp(target_step, steps, vals))


def mean_over_range(g, s, key, task, lo, hi):
    steps, vals = stitch(g, s, key, task)
    if len(vals) == 0: return float("nan")
    mask = (steps >= lo) & (steps <= hi)
    if mask.sum() == 0: return float("nan")
    return float(np.mean(vals[mask]))


def first_step_at_threshold(g, s, task, threshold):
    """First env-step where success_rate >= threshold (linear-interp between log points)."""
    steps, vals = stitch(g, s, "success_rate", task)
    if len(vals) == 0: return float("nan")
    above = vals >= threshold
    if not above.any(): return float("nan")
    idx = int(np.argmax(above))
    if idx == 0:
        return float(steps[0])
    # Linear interp between vals[idx-1] (below) and vals[idx] (above or eq)
    v0, v1 = vals[idx - 1], vals[idx]
    s0, s1 = steps[idx - 1], steps[idx]
    if v1 == v0: return float(s1)
    frac = (threshold - v0) / (v1 - v0)
    return float(s0 + frac * (s1 - s0))


def paired_t(diffs):
    n = len(diffs)
    if n < 2: return float("nan"), float("nan"), float("nan")
    m = float(np.mean(diffs))
    sd = float(np.std(diffs, ddof=1))
    if sd == 0: return m, float("inf"), float("inf")
    t = m / (sd / np.sqrt(n))
    d = m / sd
    return m, t, d


def aggregate(g, task, key):
    per_seed = []
    common = None
    for s in SEEDS:
        steps, vals = stitch(g, s, key, task)
        if len(vals) == 0: continue
        per_seed.append((steps, vals))
        if common is None or len(steps) > len(common): common = steps
    if not per_seed: return np.array([]), np.array([[]])
    interp = np.array([np.interp(common, st, v, left=v[0], right=v[-1]) for st, v in per_seed])
    return common, interp


# ============================================================
# Analysis
# ============================================================
def main():
    out = "experiments/_a2_cold_start"
    os.makedirs(out, exist_ok=True)

    # ---- (1) Early vf_loss decay log-ratio ----
    # ratio[T] = log10(vf_loss[T] / vf_loss[5M])
    EARLY_REF = 5_242_880  # first datapoint
    CHECKPOINTS = [50_000_000, 100_000_000, 200_000_000]

    lines = []
    lines.append("# A2 — Critic Cold-Start Anchoring Analysis\n")
    lines.append("Tests whether PSC provides cold-start anchor benefit (replacement hypothesis after A1 disproved variance reduction).\n")

    # Table 1: vf_loss early decay (within-reward)
    lines.append("\n## (1) Early vf_loss decay — $\\log_{10}(\\mathrm{vf\\_loss}[T] / \\mathrm{vf\\_loss}[5M])$\n")
    lines.append("More-negative = critic learning faster. **Within-reward only** (vf_loss is reward-scale-specific).\n")
    lines.append("| Task | Group | log₁₀ ratio @50M | @100M | @200M |")
    lines.append("|:---:|:---:|:---:|:---:|:---:|")
    table1: Dict = {}
    for task in ["velocity", "waypoint", "disturbance"]:
        for g in TASK_GROUPS[task]:
            ratios = {tag: [] for tag in CHECKPOINTS}
            for s in SEEDS:
                v_ref = val_at(g, s, "vf_loss", task, EARLY_REF)
                if np.isnan(v_ref) or v_ref <= 0: continue
                for tag in CHECKPOINTS:
                    v_t = val_at(g, s, "vf_loss", task, tag)
                    if np.isnan(v_t) or v_t <= 0: continue
                    ratios[tag].append(np.log10(v_t / v_ref))
            cells = []
            for tag in CHECKPOINTS:
                if len(ratios[tag]) < 2:
                    cells.append("—")
                else:
                    arr = np.array(ratios[tag])
                    cells.append(f"{arr.mean():+.3f} ± {arr.std(ddof=1):.3f}")
            table1.setdefault(task, {})[g] = ratios
            lines.append(f"| {TASK_LABEL[task]} | {g} | {cells[0]} | {cells[1]} | {cells[2]} |")

    # Within-reward paired contrasts at 50M (most discriminating early stage)
    lines.append("\n### Within-reward paired contrasts on log-ratio @50M\n")
    lines.append("(negative Δ ⇒ PSC variant decays vf_loss faster than baseline)\n")
    lines.append("| Task | Contrast | Δ log-ratio @50M | t (df=4) | d | Direction |")
    lines.append("|:---:|:---:|:---:|:---:|:---:|:---:|")
    contrasts = [
        ("velocity", "C - A", "C", "A"), ("velocity", "F - A", "F", "A"), ("velocity", "C - F", "C", "F"),
        ("velocity", "D - B", "D", "B"), ("velocity", "E - B", "E", "B"), ("velocity", "D - E", "D", "E"),
        ("waypoint", "C - A", "C", "A"), ("waypoint", "F - A", "F", "A"), ("waypoint", "C - F", "C", "F"),
        ("waypoint", "D - B", "D", "B"),
        ("disturbance", "C - A", "C", "A"), ("disturbance", "F - A", "F", "A"), ("disturbance", "C - F", "C", "F"),
    ]
    for task, lab, ga, gb in contrasts:
        if task not in table1: continue
        if ga not in table1[task] or gb not in table1[task]: continue
        tag = 50_000_000
        ra = table1[task][ga].get(tag, [])
        rb = table1[task][gb].get(tag, [])
        # paired: need both seeds with valid values
        # We re-collect per-seed alignment
        diffs = []
        for s_idx, s in enumerate(SEEDS):
            va = val_at(ga, s, "vf_loss", task, EARLY_REF)
            va_t = val_at(ga, s, "vf_loss", task, tag)
            vb = val_at(gb, s, "vf_loss", task, EARLY_REF)
            vb_t = val_at(gb, s, "vf_loss", task, tag)
            if any(np.isnan(x) or x <= 0 for x in [va, va_t, vb, vb_t]): continue
            diffs.append(np.log10(va_t / va) - np.log10(vb_t / vb))
        if len(diffs) < 2: continue
        m, t, d = paired_t(np.array(diffs))
        sig = "↓ (PSC anchors)" if m < 0 and abs(t) > 2.776 else ("↑" if m > 0 and abs(t) > 2.776 else "n.s.")
        lines.append(f"| {TASK_LABEL[task]} | {lab} | {m:+.3f} | {t:+.2f} | {d:+.2f} | {sig} |")

    # ---- (2) Early v_phys_ratio average over 5M-50M ----
    lines.append("\n## (2) Early v_phys_ratio mean over 5M–50M (PSC groups only)\n")
    lines.append("If PSC's structural prior anchors the critic during cold-start, this should be near 1.0 early.\n")
    lines.append("| Group | T1 | T2 | T3 |")
    lines.append("|:---:|:---:|:---:|:---:|")
    for g in "CDE":
        cells = []
        for task in ["velocity", "waypoint", "disturbance"]:
            vals = []
            for s in SEEDS:
                m = mean_over_range(g, s, "v_phys_ratio", task, 5_000_000, 50_000_000)
                if not np.isnan(m): vals.append(m)
            if not vals: cells.append("—")
            elif np.mean(vals) < 1e-6: cells.append("—")
            else: cells.append(f"{np.mean(vals):.3f} ± {np.std(vals, ddof=1):.3f}")
        lines.append(f"| {g} | {cells[0]} | {cells[1]} | {cells[2]} |")

    # ---- (3) Time-to-SR threshold (within and across reward) ----
    lines.append("\n## (3) Time-to-SR threshold — env-steps to first reach SR\n")
    lines.append("Reward-type-independent (SR is computed identically regardless of reward shape).\n")
    THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5]
    lines.append("| Task | Group | " + " | ".join([f"SR={t:.1f} (M steps)" for t in THRESHOLDS]) + " |")
    lines.append("|:---:|:---:|" + ":---:|" * len(THRESHOLDS))
    sr_data: Dict = {}
    for task in ["velocity", "waypoint", "disturbance"]:
        for g in TASK_GROUPS[task]:
            cells = []
            sr_data.setdefault(task, {})[g] = {}
            for th in THRESHOLDS:
                ss = []
                for s in SEEDS:
                    st = first_step_at_threshold(g, s, task, th)
                    if not np.isnan(st): ss.append(st)
                sr_data[task][g][th] = ss
                if len(ss) < 5:
                    if not ss:
                        cells.append("never")
                    else:
                        cells.append(f"{np.median(ss)/1e6:.0f} ({len(ss)}/5)")
                else:
                    cells.append(f"{np.median(ss)/1e6:.0f}")
            lines.append(f"| {TASK_LABEL[task]} | {g} | " + " | ".join(cells) + " |")

    # Within-reward speedup ratio at SR=0.2 (cold-start indicator)
    lines.append("\n### Within-reward speedup ratio at SR=0.2 (cold-start exit)\n")
    lines.append("Ratio = baseline / variant. >1.0 means PSC reaches SR=0.2 in fewer steps.\n")
    lines.append("| Task | Contrast | Median steps (a) | Median steps (b) | Speedup b/a |")
    lines.append("|:---:|:---:|:---:|:---:|:---:|")
    sr_contrasts = [
        ("velocity", "C vs A", "C", "A"), ("velocity", "F vs A", "F", "A"),
        ("velocity", "D vs B", "D", "B"), ("velocity", "E vs B", "E", "B"),
        ("waypoint", "C vs A", "C", "A"), ("waypoint", "F vs A", "F", "A"),
        ("disturbance", "C vs A", "C", "A"), ("disturbance", "F vs A", "F", "A"),
    ]
    for task, lab, ga, gb in sr_contrasts:
        if ga not in sr_data.get(task, {}) or gb not in sr_data[task]: continue
        sa = sr_data[task][ga][0.2]
        sb = sr_data[task][gb][0.2]
        if len(sa) < 3 or len(sb) < 3: continue
        ma, mb = np.median(sa), np.median(sb)
        if ma <= 0: continue
        lines.append(f"| {TASK_LABEL[task]} | {lab} | {ma/1e6:.0f}M | {mb/1e6:.0f}M | {mb/ma:.2f}× |")

    with open(f"{out}/a2_table.md", "w") as f:
        f.write("\n".join(lines))
    print(f"[+] wrote {out}/a2_table.md")

    # ---- Figures ----
    # Fig 1: early vf_loss curves, dense vs sparse, log scale, 0-200M
    fig, axes = plt.subplots(1, 2, figsize=(DBL_WIDTH, ROW_H))
    for ax, pair, title in [(axes[0], ["A", "C", "F"], "Dense reward (T1)"),
                            (axes[1], ["B", "D", "E"], "Sparse reward (T1)")]:
        for g in pair:
            steps, arr = aggregate(g, "velocity", "vf_loss")
            if arr.size == 0: continue
            mask = steps <= 200_000_000
            med = np.median(arr[:, mask], axis=0)
            q1 = np.percentile(arr[:, mask], 25, axis=0)
            q3 = np.percentile(arr[:, mask], 75, axis=0)
            ax.plot(steps[mask] / 1e6, med, label=LABELS[g], color=COLORS[g], lw=1.3)
            ax.fill_between(steps[mask] / 1e6, q1, q3, color=COLORS[g], alpha=0.15, lw=0)
        ax.set_yscale("log")
        ax.set_title(title)
        ax.set_xlabel("Env steps (M)")
        ax.set_ylabel("vf_loss (log scale)")
        ax.legend(loc="best", framealpha=0.85, edgecolor='none', fontsize=7)
        ax.set_xlim(0, 200)
    plt.tight_layout(pad=0.5)
    fig.savefig(f"{out}/a2_vfloss_early.pdf"); fig.savefig(f"{out}/a2_vfloss_early.png", dpi=160)
    plt.close(fig)
    print(f"[+] wrote {out}/a2_vfloss_early.pdf")

    # Fig 2: early v_phys_ratio for PSC groups, 0-200M
    fig, ax = plt.subplots(1, 1, figsize=(COL_WIDTH, ROW_H))
    for g in ["C", "D", "E"]:
        steps, arr = aggregate(g, "velocity", "v_phys_ratio")
        if arr.size == 0: continue
        mask = steps <= 200_000_000
        med = np.median(arr[:, mask], axis=0)
        q1 = np.percentile(arr[:, mask], 25, axis=0)
        q3 = np.percentile(arr[:, mask], 75, axis=0)
        ax.plot(steps[mask] / 1e6, med, label=LABELS[g], color=COLORS[g], lw=1.5)
        ax.fill_between(steps[mask] / 1e6, q1, q3, color=COLORS[g], alpha=0.18, lw=0)
    ax.set_title("Early $|V_{\\mathrm{phys}}|/|V_{\\mathrm{total}}|$ (T1)")
    ax.set_xlabel("Env steps (M)")
    ax.set_ylabel("Ratio")
    ax.set_xlim(0, 200)
    ax.legend(loc="best", framealpha=0.85, edgecolor='none', fontsize=8)
    plt.tight_layout(pad=0.5)
    fig.savefig(f"{out}/a2_vphys_ratio_early.pdf"); fig.savefig(f"{out}/a2_vphys_ratio_early.png", dpi=160)
    plt.close(fig)
    print(f"[+] wrote {out}/a2_vphys_ratio_early.pdf")

    # Fig 3: Time-to-SR=0.2 bars, T1 only
    fig, ax = plt.subplots(1, 1, figsize=(COL_WIDTH, ROW_H))
    x_labels = []
    medians = []
    iqr_low = []
    iqr_high = []
    colors = []
    for g in ["A", "B", "C", "D", "E", "F"]:
        if g not in sr_data["velocity"]: continue
        ss = sr_data["velocity"][g][0.2]
        if not ss:
            continue
        med = np.median(ss) / 1e6
        q1 = np.percentile(ss, 25) / 1e6
        q3 = np.percentile(ss, 75) / 1e6
        x_labels.append(g)
        medians.append(med)
        iqr_low.append(med - q1)
        iqr_high.append(q3 - med)
        colors.append(COLORS[g])
    ax.bar(range(len(medians)), medians,
           yerr=[iqr_low, iqr_high],
           color=colors, edgecolor='black', linewidth=0.4, capsize=2)
    ax.set_xticks(range(len(medians)))
    ax.set_xticklabels(x_labels)
    ax.set_ylabel("Env steps to SR=0.2 (M)")
    ax.set_title("Cold-start exit time, T1")
    plt.tight_layout(pad=0.5)
    fig.savefig(f"{out}/a2_sr_threshold.pdf"); fig.savefig(f"{out}/a2_sr_threshold.png", dpi=160)
    plt.close(fig)
    print(f"[+] wrote {out}/a2_sr_threshold.pdf")

    # ---- Verdict ----
    print()
    print("=" * 72)
    print("A2 verdict — does PSC anchor the critic in cold-start phase?")
    print("=" * 72)
    print("\n[1] vf_loss log-ratio @50M (negative ⇒ faster decay)")
    for task, lab, ga, gb in contrasts:
        diffs = []
        for s in SEEDS:
            va = val_at(ga, s, "vf_loss", task, EARLY_REF)
            va_t = val_at(ga, s, "vf_loss", task, 50_000_000)
            vb = val_at(gb, s, "vf_loss", task, EARLY_REF)
            vb_t = val_at(gb, s, "vf_loss", task, 50_000_000)
            if any(np.isnan(x) or x <= 0 for x in [va, va_t, vb, vb_t]): continue
            diffs.append(np.log10(va_t/va) - np.log10(vb_t/vb))
        if len(diffs) < 2: continue
        m, t, d = paired_t(np.array(diffs))
        verdict = "  ↓ PSC anchors faster" if m < 0 and abs(t) > 2.776 else ("  ↑ slower" if m > 0 and abs(t) > 2.776 else "  n.s.")
        print(f"  {TASK_LABEL[task]} {lab:<8} Δ={m:+.3f} t={t:+.2f} d={d:+.2f}{verdict}")

    print("\n[2] v_phys_ratio early mean over 5-50M (PSC groups, T1)")
    for g in "CDE":
        vals = []
        for s in SEEDS:
            m = mean_over_range(g, s, "v_phys_ratio", "velocity", 5_000_000, 50_000_000)
            if not np.isnan(m): vals.append(m)
        if vals:
            print(f"  {g}: {np.mean(vals):.3f} ± {np.std(vals, ddof=1):.3f}  (n={len(vals)})")

    print("\n[3] Time-to-SR=0.2 (T1, median across 5 seeds)")
    for g in "ABCDEF":
        if g not in sr_data["velocity"]: continue
        ss = sr_data["velocity"][g][0.2]
        if ss:
            print(f"  {g}: {np.median(ss)/1e6:.0f}M steps (n={len(ss)}/5)")
        else:
            print(f"  {g}: never reached SR=0.2")


if __name__ == "__main__":
    main()
