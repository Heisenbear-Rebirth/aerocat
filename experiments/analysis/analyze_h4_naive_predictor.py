"""
H4 — Naïve physical-signal baseline for crash prediction (H9 sanity check).

H9 reported V_phys AUROC = 0.80-0.83 for OOD crash prediction (K=20 steps lookahead)
in PSC groups, framing it as "interpretable deployment-time safety monitor". The
load-bearing question for that claim is: how does V_phys compare to naïve physical
signals that ANY controller exposes — bare tilt, saturation, v_err?

If naïve signals already achieve AUROC > 0.9 with tilt alone, then V_phys's value as
an interpretable monitor collapses — engineers don't need PSC for this. If V_phys
is comparable or better than naïve signals, H9's claim survives.

Setup (uses C2/D1 trajectory data, separate eval run from C1 but same 6×5×5 cells):
  Crash label = "tilt exceeds 1.5 rad within K=20 steps from t" (env-defined crash criterion)
  Predictors at time t (raw scalars):
    - tilt[t,e]           (current orientation deviation)
    - saturation[t,e]     (current actuator near-saturation flag)
    - v_err[t,e]          (current velocity tracking error magnitude)
  Compute AUROC per (group, seed, lambda).

Crucial: This baseline lets us judge whether H9's V_phys (which is a learned 5-basis
linear combination of physical quantities) outperforms hand-picked single physical
quantities. If V_phys is similar to bare tilt, then V_phys is "tilt + minor refinement".

Outputs:
  experiments/_h4_naive_predictor/
    h4_table.md
    h4_auroc_vs_lambda.{pdf,png}
    h4_per_cell.json
"""
import json
import os
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
from scipy.stats import rankdata

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
K_WINDOW = 20
TILT_CRASH = 1.5

TRAJ_DIR = "experiments/_c2d1/T1_trajectories"
OUT = "experiments/_h4_naive_predictor"


def signed_auroc(score: np.ndarray, label: np.ndarray) -> Tuple[float, str]:
    if score.size == 0 or label.sum() == 0 or label.sum() == label.size:
        return np.nan, "?"
    ranks = rankdata(score)
    n_pos = int((label == 1).sum())
    n_neg = int((label == 0).sum())
    sum_pos_ranks = float(ranks[label == 1].sum())
    auc = (sum_pos_ranks - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    if auc >= 0.5:
        return auc, "high"
    return 1.0 - auc, "low"


def process_cell(path: str, K: int = K_WINDOW) -> dict:
    z = np.load(path, allow_pickle=True)
    tilt = z["tilt"].astype(np.float32)         # (T, E)
    sat = z["saturation"].astype(np.float32)
    v_err = z["v_err"].astype(np.float32)
    active = z["active"].astype(bool)
    T, E = tilt.shape

    T_eff = T - K
    if T_eff <= 0:
        return None

    # Label: imminent tilt > 1.5 within (t, t+K]
    label = np.zeros((T_eff, E), dtype=bool)
    for e in range(E):
        exc_idx = np.where(tilt[:, e] > TILT_CRASH)[0]
        for ei in exc_idx:
            t_start = max(0, ei - K)
            t_end = min(T_eff, ei)
            if t_start >= t_end: continue
            label[t_start:t_end, e] = True

    # mask: active AND tilt[t] <= TILT_CRASH (don't predict the crash AT the crash)
    mask = active[:T_eff] & (tilt[:T_eff] <= TILT_CRASH)
    m = mask.reshape(-1)
    if m.sum() < 10:
        return None

    feats = {
        "tilt": tilt[:T_eff].reshape(-1)[m],
        "saturation": sat[:T_eff].reshape(-1)[m],
        "v_err": v_err[:T_eff].reshape(-1)[m],
        "tilt_plus_verr": (tilt[:T_eff] + 0.1 * v_err[:T_eff]).reshape(-1)[m],
    }
    y = label.reshape(-1)[m].astype(int)

    out = dict(
        n_samples=int(m.sum()),
        pos_rate=float(y.mean()) if m.sum() > 0 else np.nan,
    )
    for fname, fvals in feats.items():
        auc, direction = signed_auroc(fvals, y)
        out[f"auc_{fname}"] = auc
        out[f"dir_{fname}"] = direction
    return out


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
    raw = {}
    for g in GROUPS:
        for s in SEEDS:
            for lam in LAMBDAS:
                fn = f"{TRAJ_DIR}/traj_{g}_{s}_lam{lam}.npz"
                if not os.path.exists(fn):
                    raw[(g, s, lam)] = None
                    continue
                raw[(g, s, lam)] = process_cell(fn, K=K_WINDOW)

    dump = {f"{g}_{s}_lam{lam}": v for (g, s, lam), v in raw.items() if v is not None}
    with open(f"{OUT}/h4_per_cell.json", "w") as f:
        json.dump(dump, f, indent=2)
    print(f"[+] wrote {OUT}/h4_per_cell.json ({len(dump)} cells)")

    # ============================================================
    # Table
    # ============================================================
    lines = []
    lines.append(f"# H4 — Naïve physical-signal AUROC for crash prediction (K={K_WINDOW} steps)\n")
    lines.append(f"Label: 'tilt > {TILT_CRASH} within K=20 steps'. Predictors: bare tilt, saturation, v_err, tilt+0.1·v_err.\n")
    lines.append("**Compare to H9's V_phys AUROC** (0.80-0.83 in PSC groups at λ=1.0).\n")
    lines.append("Random = 0.5. If naïve tilt achieves > 0.95, H9's PSC-interpretability claim is weakened.\n")

    for feat in ["tilt", "saturation", "v_err", "tilt_plus_verr"]:
        lines.append(f"\n## Predictor: {feat}\n")
        lines.append("AUROC mean ± SD across 5 seeds, per (group, λ); direction in parens.\n")
        lines.append("| Group | λ=0.0 | λ=0.3 | λ=0.5 | λ=0.7 | λ=1.0 |")
        lines.append("|:---:|:---:|:---:|:---:|:---:|:---:|")
        for g in GROUPS:
            row = [g]
            for lam in LAMBDAS:
                aucs = []; dirs = []
                for s in SEEDS:
                    cell = raw.get((g, s, lam))
                    if cell is None: continue
                    a = cell.get(f"auc_{feat}", np.nan)
                    d = cell.get(f"dir_{feat}", "?")
                    if not np.isnan(a):
                        aucs.append(a); dirs.append(d)
                if len(aucs) < 2:
                    row.append("—")
                else:
                    dir_tag = max(set(dirs), key=dirs.count)
                    row.append(f"{np.mean(aucs):.3f}±{np.std(aucs, ddof=1):.3f} ({dir_tag})")
            lines.append("| " + " | ".join(row) + " |")

    # ---- Head-to-head highlight at lambda=1.0 ----
    lines.append("\n## Head-to-head AUROC at λ=1.0 (PSC groups only, K=20)\n")
    lines.append("Compare each naïve predictor's mean AUROC to H9's reported V_phys AUROC.\n")
    H9_VPHYS_LAM1 = {"C": 0.810, "D": 0.833, "E": 0.796}
    H9_VPHYS_IMM = {"C": 0.668, "D": 0.754, "E": 0.738}
    lines.append("| Group | tilt (H4) | saturation (H4) | v_err (H4) | tilt+0.1·v_err (H4) | **V_phys (H9 crash)** | **V_phys (H9 imm)** |")
    lines.append("|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")
    for g in ["C", "D", "E"]:
        cells = [g]
        for feat in ["tilt", "saturation", "v_err", "tilt_plus_verr"]:
            aucs = []
            for s in SEEDS:
                cell = raw.get((g, s, 1.0))
                if cell is None: continue
                a = cell.get(f"auc_{feat}", np.nan)
                if not np.isnan(a): aucs.append(a)
            if len(aucs) >= 2:
                cells.append(f"{np.mean(aucs):.3f}")
            else:
                cells.append("—")
        cells.append(f"**{H9_VPHYS_LAM1[g]:.3f}**")
        cells.append(f"**{H9_VPHYS_IMM[g]:.3f}**")
        lines.append("| " + " | ".join(cells) + " |")

    lines.append("\n*H4 label = tilt-exceedance within K; H9 label = done-with-reward<0 (crash) or any-done (imm). Different label definitions, similar OOD severity.*\n")

    with open(f"{OUT}/h4_table.md", "w") as f:
        f.write("\n".join(lines))
    print(f"[+] wrote {OUT}/h4_table.md")

    # ============================================================
    # Figure: AUROC vs lambda
    # ============================================================
    fig, axes = plt.subplots(1, 2, figsize=(DBL_WIDTH, ROW_H))
    for ax, feat, title in zip(axes, ["tilt", "tilt_plus_verr"],
                                ["Tilt alone as crash predictor",
                                 "Tilt + 0.1·v_err as crash predictor"]):
        for g in GROUPS:
            mus = []
            for lam in LAMBDAS:
                aucs = []
                for s in SEEDS:
                    cell = raw.get((g, s, lam))
                    if cell is None: continue
                    a = cell.get(f"auc_{feat}", np.nan)
                    if not np.isnan(a): aucs.append(a)
                mus.append(np.mean(aucs) if len(aucs) >= 2 else np.nan)
            ax.plot(LAMBDAS, mus, "-o", color=COLORS[g], label=LABELS[g], lw=1.3, ms=3)
        ax.axhline(0.5, color="black", lw=0.5, alpha=0.5)
        ax.set_xlabel("OOD severity λ")
        ax.set_ylabel("AUROC")
        ax.set_title(title, fontsize=9)
        ax.set_ylim(0.45, 1.0)
        ax.legend(loc="best", framealpha=0.85, edgecolor='none', fontsize=6)
    plt.tight_layout(pad=0.5)
    fig.savefig(f"{OUT}/h4_auroc_vs_lambda.pdf"); fig.savefig(f"{OUT}/h4_auroc_vs_lambda.png", dpi=160)
    plt.close(fig)
    print(f"[+] wrote {OUT}/h4_auroc_vs_lambda.pdf")

    # ============================================================
    # Verdict
    # ============================================================
    print()
    print("=" * 72)
    print(f"H4 verdict — do naïve physical signals beat V_phys for crash prediction?")
    print("=" * 72)
    print(f"\nAUROC at λ=1.0 (mean across 5 seeds, K={K_WINDOW}):")
    print(f"{'Group':>6}  {'tilt':>7}  {'sat':>7}  {'v_err':>7}  {'tilt+verr':>10}  {'H9 V_phys':>10}")
    for g in GROUPS:
        cells = []
        for feat in ["tilt", "saturation", "v_err", "tilt_plus_verr"]:
            aucs = []
            for s in SEEDS:
                cell = raw.get((g, s, 1.0))
                if cell is None: continue
                a = cell.get(f"auc_{feat}", np.nan)
                if not np.isnan(a): aucs.append(a)
            cells.append(np.mean(aucs) if len(aucs) >= 2 else np.nan)
        h9 = H9_VPHYS_LAM1.get(g, np.nan)
        h9_str = f"{h9:.3f}" if not np.isnan(h9) else "  N/A"
        print(f"{g:>6}  {cells[0]:>7.3f}  {cells[1]:>7.3f}  {cells[2]:>7.3f}  {cells[3]:>10.3f}  {h9_str:>10}")

    # Per-group "best naïve vs V_phys" for PSC groups
    print(f"\nFor PSC groups, max(naïve AUROC) vs H9 V_phys:")
    for g in ["C", "D", "E"]:
        cells = []
        for feat in ["tilt", "saturation", "v_err", "tilt_plus_verr"]:
            aucs = []
            for s in SEEDS:
                cell = raw.get((g, s, 1.0))
                if cell is None: continue
                a = cell.get(f"auc_{feat}", np.nan)
                if not np.isnan(a): aucs.append(a)
            cells.append((feat, np.mean(aucs) if len(aucs) >= 2 else np.nan))
        best_feat, best_auc = max(cells, key=lambda x: x[1] if not np.isnan(x[1]) else -1)
        h9 = H9_VPHYS_LAM1.get(g, np.nan)
        verdict = "V_phys ≥ naïve" if h9 >= best_auc else f"naïve ({best_feat}) wins"
        print(f"  {g}: best naïve = {best_feat} ({best_auc:.3f})   V_phys (H9) = {h9:.3f}   → {verdict}")
    print()


if __name__ == "__main__":
    main()
