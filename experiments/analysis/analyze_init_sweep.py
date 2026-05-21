"""Init-sensitivity sweep analysis (Wave 11) — compares plateaus at 500M.

Three configs:
  - Default (existing T1 D group, intermediate at 500M)
  - Uniform: w=[1,1,1,1,1], b=1.0
  - Zero:    w=[0,0,0,0,0], b=0.0

Defends against reviewer "are the 5 init values cherry-picked?"
"""
import json
import glob
import os
from typing import Dict, List, Tuple


CONFIGS = {
    "Default":  ("ablation_D_sparse_psc",                 "principled (45,2,2,0.5,1; b=20)"),
    "Uniform":  ("ablation_D_sparse_psc_initUniform",     "uniform (1,1,1,1,1; b=1.0)"),
    "Zero":     ("ablation_D_sparse_psc_initZero",        "zero (0,0,0,0,0; b=0.0)"),
}
SEEDS = [42, 123, 456, 789, 1024]
BASE = "experiments"


def all_metrics_paths(dir_name: str, seed: int) -> List[str]:
    return sorted(glob.glob(f"{BASE}/{dir_name}/seed_{seed}/*/*/metrics.json"))


def stitch(dir_name: str, seed: int, key: str) -> Tuple[List[int], List[float]]:
    pairs = []
    for p in all_metrics_paths(dir_name, seed):
        try:
            d = json.load(open(p))
        except (FileNotFoundError, json.JSONDecodeError):
            continue
        if key not in d:
            continue
        e = d[key]
        steps = e.get("steps", []) if isinstance(e, dict) else list(range(len(e)))
        vals = e["values"] if isinstance(e, dict) else e
        for st, v in zip(steps, vals):
            pairs.append((st, v))
    seen = set(); uniq = []
    for st, v in sorted(pairs, key=lambda x: x[0]):
        if st not in seen:
            seen.add(st); uniq.append((st, v))
    return [x[0] for x in uniq], [x[1] for x in uniq]


def plat_at_500M(dir_name: str, seed: int, key: str = "mean_reward",
                 frac: float = 0.1, target: int = 500_000_000) -> float:
    """Plateau = last 10% mean of trajectory truncated at <= 500M steps."""
    sts, vs = stitch(dir_name, seed, key)
    if not vs:
        return 0.0
    # truncate at <= 500M
    pairs = [(st, v) for st, v in zip(sts, vs) if st <= target]
    if not pairs:
        return 0.0
    vs_trunc = [v for _, v in pairs]
    n = max(1, int(len(vs_trunc) * frac))
    return sum(vs_trunc[-n:]) / n


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


# =============================================================================
print("=" * 95)
print("AeroCat v19.4 — PSC Init-Sensitivity Sweep Analysis (Wave 11)")
print("=" * 95)
print()
print(f"  metric: mean_reward, plateau = last 10% of trajectory truncated at 500M steps")
print()

# ----- per-(config, seed) ----
print("§1  Per-(config, seed) plateau @ 500M, mean_reward")
print()
hdr = f"  {'config':>10s}  " + "  ".join(f"{f'seed={s}':>10s}" for s in SEEDS) + f"  {'mean':>9s}  {'sd':>9s}"
print(hdr)
print("  " + "-" * (len(hdr) - 2))

plat: Dict[str, List[float]] = {}
for label, (dir_name, _) in CONFIGS.items():
    vs = [plat_at_500M(dir_name, s) for s in SEEDS]
    plat[label] = vs
    cells = "  ".join(f"{v:>10.4f}" for v in vs)
    print(f"  {label:>10s}  {cells}  {mu(vs):>9.4f}  {sd(vs):>9.4f}")

# ----- contrasts vs Default ----
print()
print("§2  Paired contrasts vs Default (principled init)")
print()
for alt in ["Uniform", "Zero"]:
    diffs = [plat[alt][i] - plat["Default"][i] for i in range(5)]
    t, d = t_d(diffs)
    pct = (mu(diffs) / mu(plat["Default"])) * 100 if mu(plat["Default"]) != 0 else 0
    print(f"  {alt} - Default:")
    print(f"    per-seed = {[round(x, 4) for x in diffs]}")
    print(f"    mean = {mu(diffs):+.4f}   sd = {sd(diffs):.4f}   t = {t:.3f}   d = {d:.3f}   {sig(t)}")
    print(f"    relative = {pct:+.2f}% of Default plateau ({mu(plat['Default']):.4f})")
    print()

# ----- summary ----
print("=" * 95)
print("§3  Summary")
print("=" * 95)
print()
default_mean = mu(plat["Default"])
print(f"  Default plateau:   {default_mean:.4f}")
print(f"  Uniform plateau:   {mu(plat['Uniform']):.4f}  ({(mu(plat['Uniform'])/default_mean - 1)*100:+.2f}%)")
print(f"  Zero plateau:      {mu(plat['Zero']):.4f}  ({(mu(plat['Zero'])/default_mean - 1)*100:+.2f}%)")
print()

# Robustness verdict
def in_band(alt, default, band=0.10):
    return abs(mu(plat[alt]) - default) / default < band if default != 0 else False

verdict_uniform = "✓ within ±10% — INIT-INSENSITIVE" if in_band("Uniform", default_mean) else "⚠ exceeds ±10%"
verdict_zero    = "✓ within ±10% — INIT-INSENSITIVE" if in_band("Zero", default_mean) else "⚠ exceeds ±10%"
print(f"  Uniform: {verdict_uniform}")
print(f"  Zero:    {verdict_zero}")
print()

# ----- generate plot ----
try:
    import matplotlib.pyplot as plt
    import numpy as np

    out_dir = "experiments/_init_analysis"
    os.makedirs(out_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    labels = list(CONFIGS.keys())
    means = [mu(plat[l]) for l in labels]
    sds = [sd(plat[l]) for l in labels]
    colors = ["#22aa44", "#3377cc", "#cc7733"]
    bars = ax.bar(labels, means, yerr=sds, color=colors, capsize=8,
                  error_kw={"linewidth": 1.0})
    for b, m, s in zip(bars, means, sds):
        ax.text(b.get_x() + b.get_width()/2, m + s + 0.02,
                f"{m:.3f}\n±{s:.3f}", ha="center", va="bottom", fontsize=9)

    # ±10% band of default
    ax.axhline(default_mean, color="black", lw=0.8, ls="--", alpha=0.5,
               label=f"Default plateau ({default_mean:.3f})")
    ax.axhspan(default_mean*0.9, default_mean*1.1, color="gray", alpha=0.12,
               label=r"±10% band")

    ax.set_ylabel("Plateau mean_reward (last 10% @ 500M steps)")
    ax.set_title("PSC initialization-sensitivity sweep (Group D, 5 seeds × 500M)")
    ax.grid(True, axis='y', alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/init_sensitivity_bars.png", dpi=120)
    plt.close()
    print(f"    [+] {out_dir}/init_sensitivity_bars.png")

except Exception as e:
    print(f"    [!] plot failed: {e}")

print()
print("=" * 95)
print("Done.")
