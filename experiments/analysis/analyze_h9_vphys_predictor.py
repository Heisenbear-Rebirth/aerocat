"""
H9 — V_phys as deployment-time crash predictor (AUROC analysis).

Hypothesis: V_phys(s_t), being an analytic function of physical state (5 hand-picked
basis × scalar weights), is an _interpretable_ predictor of imminent crash/termination.
If V_phys's AUROC > random and comparable to V_total's, then V_phys can serve as an
explainable safety monitor at deployment time — the basis activations decompose the
signal into named physical quantities (vel_err, ang_vel, tilt, PID_int, saturation),
unlike opaque V_total.

Setup (per group × seed × lambda, T=500 steps × E=256 envs):
  For each (t, e) with active[t,e]=1 and t+K ≤ T-1:
    label_imm_done = any(done[t+1:t+K+1, e])              "imminent termination within K"
    label_crash    = label_imm_done AND (rew at that done step < 0)  "imminent crash"
  Features (signed; AUROC handles direction):
    V_total[t,e]
    V_phys[t,e]
    V_res[t,e]
  K = 20 steps (≈ 0.4–1 sec lookahead depending on dt)

Per (group, lam, seed), report AUROC for V_phys, V_res, V_total predicting label_imm_done
and label_crash.

Outputs:
  experiments/_h9_vphys_predictor/
    h9_table.md
    h9_auroc_vs_lambda.{pdf,png}     AUROC of V_phys vs V_total, per group
    h9_per_cell.json
"""
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
K_WINDOW = 20  # steps lookahead for "imminent" label

TRAJ_DIR = "experiments/_c1/T1_calib_trajectories"
OUT = "experiments/_h9_vphys_predictor"


def manual_auroc(score: np.ndarray, label: np.ndarray) -> float:
    """Compute AUROC. Returns max(AUC, 1-AUC) so it's always >= 0.5 (we don't
    pre-commit to direction). Returns NaN if labels are constant.
    score, label flat arrays of equal length."""
    if score.size == 0:
        return np.nan
    if label.sum() == 0 or label.sum() == label.size:
        return np.nan
    # Mann-Whitney U statistic, normalized
    # Rank all scores, then AUC = (mean rank of positives - (n_pos+1)/2) / n_neg
    pos = score[label == 1]
    neg = score[label == 0]
    n_pos = pos.size
    n_neg = neg.size
    # Vectorized: use argsort ranks
    ranks = np.empty_like(score, dtype=np.float64)
    order = np.argsort(score, kind="stable")
    ranks[order] = np.arange(1, score.size + 1)
    # Tie correction: average ranks for tied values
    # Use simple approach: scipy.stats.rankdata
    from scipy.stats import rankdata
    ranks = rankdata(score)
    sum_pos_ranks = float(ranks[label == 1].sum())
    auc = (sum_pos_ranks - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    # Return un-folded (predictor direction is informative)
    # But report max(AUC, 1-AUC) for "discrimination strength" interpretation
    return float(max(auc, 1.0 - auc))


def signed_auroc(score: np.ndarray, label: np.ndarray) -> Tuple[float, str]:
    """Returns (AUC >= 0.5, "low" or "high") for direction."""
    if score.size == 0:
        return np.nan, "?"
    if label.sum() == 0 or label.sum() == label.size:
        return np.nan, "?"
    from scipy.stats import rankdata
    ranks = rankdata(score)
    n_pos = int((label == 1).sum())
    n_neg = int((label == 0).sum())
    sum_pos_ranks = float(ranks[label == 1].sum())
    auc = (sum_pos_ranks - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    if auc >= 0.5:
        return auc, "high"  # positives have higher score
    else:
        return 1.0 - auc, "low"  # positives have lower score (more predictive when score is low)


def process_cell(path: str, K: int = K_WINDOW) -> dict:
    z = np.load(path)
    v = z["value"].astype(np.float64)
    vp = z["v_phys"].astype(np.float64)
    vr = z["v_res"].astype(np.float64)
    reward = z["reward"].astype(np.float64)
    done = z["done"]
    active = z["active"].astype(bool)
    T, E = done.shape

    # For each (t, e), label = any done in (t, t+K]
    # Build label_imm and label_crash arrays of shape (T-K, E) — only over feasible t
    T_eff = T - K
    if T_eff <= 0:
        return None
    label_imm = np.zeros((T_eff, E), dtype=bool)
    label_crash = np.zeros((T_eff, E), dtype=bool)
    for e in range(E):
        d_idx = np.where(done[:, e] == 1)[0]
        if len(d_idx) == 0:
            continue
        for di in d_idx:
            r_at = reward[di, e]
            t_start = max(0, di - K)
            t_end = min(T_eff, di)
            if t_start >= t_end:
                continue
            label_imm[t_start:t_end, e] = True
            if r_at < 0:
                label_crash[t_start:t_end, e] = True

    # Mask: active and not-currently-done at t
    mask = active[:T_eff] & (~done[:T_eff])

    # Flatten
    m = mask.reshape(-1)
    v_flat = v[:T_eff].reshape(-1)[m]
    vp_flat = vp[:T_eff].reshape(-1)[m]
    vr_flat = vr[:T_eff].reshape(-1)[m]
    yimm = label_imm.reshape(-1)[m]
    ycra = label_crash.reshape(-1)[m]

    out = dict(
        n_samples=int(m.sum()),
        pos_rate_imm=float(yimm.mean()) if m.sum() > 0 else np.nan,
        pos_rate_crash=float(ycra.mean()) if m.sum() > 0 else np.nan,
    )

    for label_name, y in [("imm", yimm), ("crash", ycra)]:
        for feat_name, feat in [("V", v_flat), ("V_phys", vp_flat), ("V_res", vr_flat)]:
            auc, direction = signed_auroc(feat, y.astype(int))
            out[f"auc_{feat_name}_{label_name}"] = auc
            out[f"dir_{feat_name}_{label_name}"] = direction
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

    # ---- Collect per-cell metrics ----
    raw: Dict[Tuple[str, int, float], dict] = {}
    for g in ALL_GROUPS:
        for s in SEEDS:
            for lam in LAMBDAS:
                fn = f"{TRAJ_DIR}/traj_{g}_{s}_lam{lam}.npz"
                if not os.path.exists(fn):
                    raw[(g, s, lam)] = None
                    continue
                raw[(g, s, lam)] = process_cell(fn, K=K_WINDOW)

    # ---- Per-cell JSON dump ----
    dump = {}
    for (g, s, lam), v in raw.items():
        if v is None: continue
        dump[f"{g}_{s}_lam{lam}"] = v
    with open(f"{OUT}/h9_per_cell.json", "w") as f:
        json.dump(dump, f, indent=2)
    print(f"[+] wrote {OUT}/h9_per_cell.json ({len(dump)} cells)")

    # ---- Tables ----
    lines = []
    lines.append(f"# H9 — V_phys as deployment-time crash predictor (AUROC, K={K_WINDOW} steps)\n")
    lines.append("Reports AUROC ≥ 0.5 (after auto-direction flip; direction tag indicates predictive sign).\n")
    lines.append("Random guess = 0.5. AUROC > 0.7 = useful predictor; > 0.8 = strong.\n")
    lines.append(f"Crash label uses reward < 0 at done step as crash criterion (heuristic).\n")
    lines.append(f"Positive-class rate per cell ranges 0.1–5%; AUROC handles class imbalance.\n")

    for label_kind, label_desc in [("imm", "imminent ANY termination within K=20 steps"),
                                    ("crash", "imminent CRASH (reward<0 at done) within K=20 steps")]:
        lines.append(f"\n## Predicting: {label_desc}\n")
        lines.append(f"### AUROC mean ± SD across 5 seeds, per (group, λ)\n")
        for feat in ["V_phys", "V_res", "V"]:
            lines.append(f"\n#### Predictor: {feat}\n")
            lines.append("| Group | λ=0.0 | λ=0.3 | λ=0.5 | λ=0.7 | λ=1.0 |")
            lines.append("|:---:|:---:|:---:|:---:|:---:|:---:|")
            for g in ALL_GROUPS:
                # MLP groups (A/B/F) have V_phys=0 so V_phys AUROC is undefined; skip
                if feat == "V_phys" and g not in PSC_GROUPS:
                    continue
                if feat == "V_res" and g not in PSC_GROUPS:
                    # for MLP groups, V_res = V_total, redundant
                    continue
                row = [g]
                for lam in LAMBDAS:
                    aucs = []
                    dirs = []
                    for s in SEEDS:
                        cell = raw.get((g, s, lam))
                        if cell is None: continue
                        a = cell.get(f"auc_{feat}_{label_kind}", np.nan)
                        d = cell.get(f"dir_{feat}_{label_kind}", "?")
                        if not np.isnan(a):
                            aucs.append(a); dirs.append(d)
                    if len(aucs) < 2:
                        row.append("—")
                    else:
                        # majority direction
                        dir_tag = max(set(dirs), key=dirs.count)
                        row.append(f"{np.mean(aucs):.3f}±{np.std(aucs, ddof=1):.3f} ({dir_tag})")
                lines.append("| " + " | ".join(row) + " |")

        # Paired contrast: V_phys vs V_total within PSC groups
        lines.append(f"\n### Paired Δ AUROC = AUROC[V_phys] − AUROC[V_total] within PSC groups (5 seeds, label={label_kind})\n")
        lines.append("Positive ⇒ V_phys alone is a better/comparable predictor than full V (interpretable substitute).\n")
        lines.append("Negative ⇒ V_total beats V_phys (V_phys adds noise relative to V_res).\n")
        lines.append("| Group | λ | Mean Δ | t (df=4) | n | Direction |")
        lines.append("|:---:|:---:|:---:|:---:|:---:|:---:|")
        for g in PSC_GROUPS:
            for lam in LAMBDAS:
                diffs = []
                for s in SEEDS:
                    cell = raw.get((g, s, lam))
                    if cell is None: continue
                    a_vp = cell.get(f"auc_V_phys_{label_kind}", np.nan)
                    a_v = cell.get(f"auc_V_{label_kind}", np.nan)
                    if np.isnan(a_vp) or np.isnan(a_v): continue
                    diffs.append(a_vp - a_v)
                m, t, d, n = paired_t(diffs)
                if np.isnan(m): continue
                if abs(t) > 2.776:
                    verd = "↑ V_phys ≥ V" if m >= 0 else "↓ V_phys < V"
                elif abs(t) > 2.132:
                    verd = "trend (p<0.10)"
                else:
                    verd = "n.s."
                lines.append(f"| {g} | {lam} | {m:+.3f} | {t:+.2f} | {n} | {verd} |")

    # ---- Stronger contrast: pooled AUC at λ=1.0 (OOD-extreme) ----
    lines.append(f"\n## Highlights: AUROC at λ=1.0 (OOD-extreme, K={K_WINDOW})\n")
    lines.append("Single-number summary for the harshest OOD condition.\n")
    lines.append("| Group | Predictor | label_imm AUROC (mean±SD) | label_crash AUROC (mean±SD) | n_seeds |")
    lines.append("|:---:|:---:|:---:|:---:|:---:|")
    for g in PSC_GROUPS:
        for feat in ["V_phys", "V_res", "V"]:
            for label_kind in ["imm", "crash"]:
                pass  # consolidate below
        # one row per (group, predictor) with both label columns
        for feat in ["V_phys", "V_res", "V"]:
            row = [g, feat]
            for label_kind in ["imm", "crash"]:
                aucs = []; dirs = []
                for s in SEEDS:
                    cell = raw.get((g, s, 1.0))
                    if cell is None: continue
                    a = cell.get(f"auc_{feat}_{label_kind}", np.nan)
                    d = cell.get(f"dir_{feat}_{label_kind}", "?")
                    if not np.isnan(a):
                        aucs.append(a); dirs.append(d)
                if len(aucs) < 2:
                    row.append("—")
                else:
                    dir_tag = max(set(dirs), key=dirs.count)
                    row.append(f"{np.mean(aucs):.3f}±{np.std(aucs, ddof=1):.3f} ({dir_tag})")
            row.append(str(len(aucs)))
            lines.append("| " + " | ".join(row) + " |")

    with open(f"{OUT}/h9_table.md", "w") as f:
        f.write("\n".join(lines))
    print(f"[+] wrote {OUT}/h9_table.md")

    # ============================================================
    # Figure: AUROC vs lambda for V_phys (PSC groups)
    # ============================================================
    fig, axes = plt.subplots(1, 2, figsize=(DBL_WIDTH, ROW_H))
    for ax, label_kind, title in zip(axes,
                                      ["imm", "crash"],
                                      [f"Imminent done within K={K_WINDOW}",
                                       f"Imminent CRASH within K={K_WINDOW}"]):
        for g in PSC_GROUPS:
            for feat, ls, label_suffix in [("V_phys", "-", "V_phys"), ("V", "--", "V_total")]:
                mus = []
                for lam in LAMBDAS:
                    aucs = []
                    for s in SEEDS:
                        cell = raw.get((g, s, lam))
                        if cell is None: continue
                        a = cell.get(f"auc_{feat}_{label_kind}", np.nan)
                        if not np.isnan(a): aucs.append(a)
                    mus.append(np.mean(aucs) if len(aucs) >= 2 else np.nan)
                ax.plot(LAMBDAS, mus, ls + "o", color=COLORS[g],
                        label=f"{LABELS[g]} · {label_suffix}", lw=1.2, ms=3)
        ax.axhline(0.5, color="black", lw=0.5, alpha=0.5)
        ax.set_xlabel("OOD severity λ")
        ax.set_ylabel("AUROC (folded ≥ 0.5)")
        ax.set_title(title, fontsize=9)
        ax.set_ylim(0.45, 1.0)
        ax.legend(loc="best", framealpha=0.85, edgecolor='none', fontsize=6)
    plt.tight_layout(pad=0.5)
    fig.savefig(f"{OUT}/h9_auroc_vs_lambda.pdf"); fig.savefig(f"{OUT}/h9_auroc_vs_lambda.png", dpi=160)
    plt.close(fig)
    print(f"[+] wrote {OUT}/h9_auroc_vs_lambda.pdf")

    # ============================================================
    # Verdict
    # ============================================================
    print()
    print("=" * 72)
    print(f"H9 verdict — does V_phys predict imminent crash (K={K_WINDOW})?")
    print("=" * 72)
    for label_kind in ["imm", "crash"]:
        print(f"\n[{label_kind}] AUROC at λ=1.0, mean across 5 seeds:")
        for g in PSC_GROUPS:
            for feat in ["V_phys", "V_res", "V"]:
                aucs = []
                for s in SEEDS:
                    cell = raw.get((g, s, 1.0))
                    if cell is None: continue
                    a = cell.get(f"auc_{feat}_{label_kind}", np.nan)
                    if not np.isnan(a): aucs.append(a)
                if len(aucs) >= 2:
                    print(f"  {g} {feat:7s} AUC={np.mean(aucs):.3f} (n={len(aucs)})")
        # Comparison V_phys vs V for each PSC group at λ=1
        for g in PSC_GROUPS:
            diffs = []
            for s in SEEDS:
                cell = raw.get((g, s, 1.0))
                if cell is None: continue
                a_vp = cell.get(f"auc_V_phys_{label_kind}", np.nan)
                a_v = cell.get(f"auc_V_{label_kind}", np.nan)
                if np.isnan(a_vp) or np.isnan(a_v): continue
                diffs.append(a_vp - a_v)
            m, t, d, n = paired_t(diffs)
            if not np.isnan(m):
                print(f"  {g} ΔAUC(V_phys-V) λ=1: {m:+.3f}  t={t:+.2f}  n={n}")
    print()


if __name__ == "__main__":
    main()
