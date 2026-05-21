"""DR/OOD eval analysis — robustness curves across lambda.

Loads experiments/_dr_eval_results.json (150 cells: 6 groups x 5 seeds x 5 lambdas)
and produces:
  1. Console table: per-(group, lambda) mean +/- SD success_rate, crash_rate, return
  2. Paired contrasts at each lambda (D-A, D-F, C-A, F-A, E-B)
  3. PNG plots: success_rate vs lambda for each group (5-seed median + IQR)
  4. PNG plot: crash_rate vs lambda

Output: experiments/_dr_analysis/
"""
import json
import os
from collections import defaultdict
from typing import Dict, List


RESULTS = "experiments/_dr_eval_results.json"
OUT = "experiments/_dr_analysis"
os.makedirs(OUT, exist_ok=True)


def mu(xs):
    return sum(xs) / len(xs) if xs else 0.0

def sd(xs):
    if len(xs) < 2: return 0.0
    m = mu(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5

def t_d(diffs):
    n = len(diffs)
    m = mu(diffs); s = sd(diffs)
    if s == 0 or n == 0: return float("inf"), 0.0
    return m / (s / (n ** 0.5)), m / s

def sig(t):
    a = abs(t)
    if a > 4.6: return "***"
    if a > 3.5: return "**"
    if a > 2.78: return "*"
    return "n.s."


# Load
data = json.load(open(RESULTS))
print(f"[+] Loaded {len(data)} cells from {RESULTS}")

# Index: dict[(group, lambda)] -> list of {seed: stat}
by_g_l: Dict[tuple, Dict[int, dict]] = defaultdict(dict)
for row in data:
    g = row["group"]; lam = row["lambda"]; s = row["seed"]
    by_g_l[(g, lam)][s] = row

GROUPS = sorted({r["group"] for r in data})
LAMBDAS = sorted({r["lambda"] for r in data})
SEEDS = sorted({r["seed"] for r in data})
print(f"  groups={GROUPS}  lambdas={LAMBDAS}  seeds={SEEDS}")

# =============================================================================
print()
print("=" * 95)
print("§1  Per-(group, lambda) success_rate  [5-seed mean +/- SD]")
print("=" * 95)
print()
hdr = f"  {'group':>5s}  " + "  ".join(f"λ={l:.1f}".rjust(14) for l in LAMBDAS)
print(hdr)
print("  " + "-" * (len(hdr) - 2))
sr: Dict[str, Dict[float, List[float]]] = {g: {} for g in GROUPS}
for g in GROUPS:
    cells = []
    for lam in LAMBDAS:
        vs = [by_g_l[(g, lam)][s]["success_rate"] for s in SEEDS if s in by_g_l[(g, lam)]]
        sr[g][lam] = vs
        if vs:
            cells.append(f"{mu(vs):.3f}±{sd(vs):.3f}")
        else:
            cells.append("    n/a    ")
    print(f"  {g:>5s}  " + "  ".join(c.rjust(14) for c in cells))

# Crash rate
print()
print("=" * 95)
print("§2  Per-(group, lambda) crash_rate  [5-seed mean +/- SD]")
print("=" * 95)
print()
print(hdr)
print("  " + "-" * (len(hdr) - 2))
cr: Dict[str, Dict[float, List[float]]] = {g: {} for g in GROUPS}
for g in GROUPS:
    cells = []
    for lam in LAMBDAS:
        vs = [by_g_l[(g, lam)][s]["crash_rate"] for s in SEEDS if s in by_g_l[(g, lam)]]
        cr[g][lam] = vs
        if vs:
            cells.append(f"{mu(vs):.3f}±{sd(vs):.3f}")
        else:
            cells.append("    n/a    ")
    print(f"  {g:>5s}  " + "  ".join(c.rjust(14) for c in cells))

# =============================================================================
print()
print("=" * 95)
print("§3  Paired contrasts at each lambda (success_rate)")
print("=" * 95)
contrasts = [
    ("D - A", "D", "A"),  # our method vs MLP-dense
    ("D - F", "D", "F"),  # our method vs Cai 2025 SOTA
    ("D - B", "D", "B"),  # our sparse vs MLP-sparse
    ("C - A", "C", "A"),  # PSC-dense vs MLP-dense
    ("E - B", "E", "B"),  # PSC-fixed-w (PBRS) vs MLP-sparse
    ("F - A", "F", "A"),  # Cai vs MLP-dense
]
print()
print(f"  {'contrast':>10s}  " + "  ".join(f"λ={l:.1f}".rjust(20) for l in LAMBDAS))
print("  " + "-" * (10 + 2 + 22 * len(LAMBDAS)))
for label, ga, gb in contrasts:
    cells = []
    for lam in LAMBDAS:
        a_vs = sr[ga].get(lam, [])
        b_vs = sr[gb].get(lam, [])
        if not a_vs or not b_vs or len(a_vs) != len(b_vs):
            cells.append("       n/a       ")
            continue
        diffs = [a_vs[i] - b_vs[i] for i in range(len(a_vs))]
        m = mu(diffs); t, d = t_d(diffs)
        cells.append(f"{m:+.3f} t={t:5.2f} {sig(t):>4s}")
    print(f"  {label:>10s}  " + "  ".join(c.rjust(20) for c in cells))

# =============================================================================
print()
print("=" * 95)
print("§4  Plots")
print("=" * 95)
try:
    import matplotlib.pyplot as plt
    import numpy as np

    color = {"A": "#888888", "B": "#cc4444", "C": "#3377cc",
             "D": "#22aa44", "E": "#aa44aa", "F": "#dd8800"}
    label_map = {"A": "A: MLP+dense", "B": "B: MLP+sparse", "C": "C: PSC+dense",
                 "D": "D: PSC+sparse (ours)", "E": "E: PSC fix-w + sparse", "F": "F: Cai 2025"}

    # Plot 1: success_rate vs lambda
    fig, ax = plt.subplots(figsize=(8, 5))
    for g in GROUPS:
        means = [mu(sr[g][l]) for l in LAMBDAS]
        sds = [sd(sr[g][l]) for l in LAMBDAS]
        ax.errorbar(LAMBDAS, means, yerr=sds, label=label_map.get(g, g),
                    color=color.get(g, "#000000"), marker="o", capsize=3, lw=1.6)
    ax.set_xlabel(r"curriculum $\lambda$ (DR strength)")
    ax.set_ylabel("success_rate (100-episode mean)")
    ax.set_title("OOD robustness curves — success_rate vs DR lambda")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{OUT}/dr_success_curves.png", dpi=120)
    plt.close()
    print(f"    [+] {OUT}/dr_success_curves.png")

    # Plot 2: crash_rate vs lambda
    fig, ax = plt.subplots(figsize=(8, 5))
    for g in GROUPS:
        means = [mu(cr[g][l]) for l in LAMBDAS]
        sds = [sd(cr[g][l]) for l in LAMBDAS]
        ax.errorbar(LAMBDAS, means, yerr=sds, label=label_map.get(g, g),
                    color=color.get(g, "#000000"), marker="s", capsize=3, lw=1.6)
    ax.set_xlabel(r"curriculum $\lambda$ (DR strength)")
    ax.set_ylabel("crash_rate (fraction of episodes)")
    ax.set_title("OOD robustness curves — crash_rate vs DR lambda")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{OUT}/dr_crash_curves.png", dpi=120)
    plt.close()
    print(f"    [+] {OUT}/dr_crash_curves.png")

    # Plot 3: tracking_rmse vs lambda
    rmse = {g: {l: [by_g_l[(g, l)][s]["tracking_rmse"] for s in SEEDS if s in by_g_l[(g, l)]]
                for l in LAMBDAS} for g in GROUPS}
    fig, ax = plt.subplots(figsize=(8, 5))
    for g in GROUPS:
        means = [mu(rmse[g][l]) for l in LAMBDAS]
        sds = [sd(rmse[g][l]) for l in LAMBDAS]
        ax.errorbar(LAMBDAS, means, yerr=sds, label=label_map.get(g, g),
                    color=color.get(g, "#000000"), marker="^", capsize=3, lw=1.6)
    ax.set_xlabel(r"curriculum $\lambda$ (DR strength)")
    ax.set_ylabel("tracking_rmse (m/s)")
    ax.set_title("OOD robustness curves — tracking_rmse vs DR lambda")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{OUT}/dr_rmse_curves.png", dpi=120)
    plt.close()
    print(f"    [+] {OUT}/dr_rmse_curves.png")

except Exception as e:
    print(f"    [!] plot failed: {e}")
    import traceback; traceback.print_exc()

print()
print("=" * 95)
print("Done.")
