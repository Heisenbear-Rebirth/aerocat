"""
Analyze E group (PSC + sparse + FIXED w_i) vs D group (PSC + sparse + LEARNABLE w_i).

This empirically tests the C1(b) PSC ≠ PBRS claim:
  - If E ≈ B (no advantage over MLP): fixed-w PSC behaves like PBRS, no improvement
  - If E < D significantly: learnable w_i is what gives PSC its power, NOT just the structure
  - If E ≈ D: structure alone matters, learnability is irrelevant (would weaken the paper's claim)

The expected outcome for a strong paper: D > E > B.
"""
import json, glob, statistics

G = {
    "B": ("sparse", "mlp"),
    "D": ("sparse", "psc"),
    "E": ("sparse", "psc_fixedw"),
}
SEEDS = [42, 123, 456, 789, 1024]


def all_paths(g, s):
    rt, ct = G[g]
    return sorted(glob.glob(f"experiments/ablation_{g}_{rt}_{ct}/seed_{s}/*/*/metrics.json"))


def stitch(g, s, k="mean_reward"):
    pairs = []
    for p in all_paths(g, s):
        d = json.load(open(p))
        if k in d:
            e = d[k]
            for st, v in zip(e.get("steps", []), e.get("values", [])):
                pairs.append((st, v))
    seen = set()
    out = []
    for st, v in sorted(pairs, key=lambda x: x[0]):
        if st not in seen:
            seen.add(st)
            out.append((st, v))
    return [x[0] for x in out], [x[1] for x in out]


def plat(g, s, k="mean_reward", frac=0.1):
    _, vals = stitch(g, s, k)
    if not vals:
        return 0.0
    n = max(1, int(len(vals) * frac))
    return sum(vals[-n:]) / n


def mu(xs):
    return sum(xs) / len(xs)


def sd(xs):
    m = mu(xs)
    return (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5


def t_stat(diffs):
    n = len(diffs)
    m = mu(diffs)
    s = sd(diffs)
    if s == 0:
        return float("inf"), 0.0
    return m / (s / (n ** 0.5)), m / s


print("=" * 90)
print("AeroCat v19.2 — PSC ≠ PBRS Empirical Test (E group vs D group vs B group)")
print("=" * 90)
print()

# Per-seed plateau values
ps = {}
for g in "BDE":
    ps[g] = [plat(g, s) for s in SEEDS]

print("§1 1B plateau values (sparse reward groups, 5 seeds)")
print()
print(f"{'group':>6s} " + "  ".join(f"seed={s:>5d}" for s in SEEDS) + f"  {'mean':>10s}  {'sd':>8s}")
for g in "BDE":
    cells = "  ".join(f"{v:>10.4f}" for v in ps[g])
    print(f"  {g:>4s}  {cells}  {mu(ps[g]):>10.4f}  {sd(ps[g]):>8.4f}")

print()
print("§2 Critical contrasts — does E differ from B and D?")
print()

# D - E: if positive, learnable w_i provides extra benefit beyond fixed structure
de = [ps["D"][i] - ps["E"][i] for i in range(5)]
t_de, d_de = t_stat(de)
print(f"  D - E (learnable vs fixed PSC weights):")
print(f"    per-seed = {[round(x,4) for x in de]}")
print(f"    mean = {mu(de):+.4f}   sd = {sd(de):.4f}   t = {t_de:.3f}   d = {d_de:.3f}")
print(f"    interpretation: {'✅ learnable matters (PSC > PBRS)' if t_de > 2.0 else '⚠ structure alone may suffice (weakens C1(b))'}")

# E - B: if positive, fixed-w PSC still beats MLP (i.e., structure helps but learnable is bonus)
eb = [ps["E"][i] - ps["B"][i] for i in range(5)]
t_eb, d_eb = t_stat(eb)
print()
print(f"  E - B (fixed-w PSC vs MLP):")
print(f"    per-seed = {[round(x,4) for x in eb]}")
print(f"    mean = {mu(eb):+.4f}   sd = {sd(eb):.4f}   t = {t_eb:.3f}   d = {d_eb:.3f}")
print(f"    interpretation: {'✅ structure alone helps' if t_eb > 2.0 else '⚠ structure ≈ MLP (PSC needs learnability)'}")

# D - B: total PSC vs no PSC
db = [ps["D"][i] - ps["B"][i] for i in range(5)]
t_db, d_db = t_stat(db)
print()
print(f"  D - B (full PSC vs MLP, reference):")
print(f"    mean = {mu(db):+.4f}   sd = {sd(db):.4f}   t = {t_db:.3f}   d = {d_db:.3f}")

# Decompose: D-B = (D-E) + (E-B)
print()
print("  Decomposition: D-B = (D-E) + (E-B)")
print(f"    {mu(db):+.4f}  =  {mu(de):+.4f} (learnable contribution)  +  {mu(eb):+.4f} (structure contribution)")
de_share = abs(mu(de)) / max(abs(mu(db)), 1e-9)
eb_share = abs(mu(eb)) / max(abs(mu(db)), 1e-9)
print(f"    learnable contributes ~{de_share*100:.0f}%, structure contributes ~{eb_share*100:.0f}%")

# PSC weight evolution check: did E group's weights actually stay fixed?
print()
print("§3 Sanity check: did E group weights stay frozen?")
print()
print(f"{'seed':>6s}  {'w0_init':>10s} {'w0_end':>10s}  {'w1_init':>10s} {'w1_end':>10s}  {'b_init':>10s} {'b_end':>10s}")
for s in SEEDS:
    paths = all_paths("E", s)
    if paths:
        d = json.load(open(paths[0]))  # first metrics file (start)
        w0_first = d["psc_w0_vel"]["values"][0] if "psc_w0_vel" in d else float("nan")
        w1_first = d["psc_w1_ang"]["values"][0] if "psc_w1_ang" in d else float("nan")
        b_first = d["psc_bias"]["values"][0] if "psc_bias" in d else float("nan")
        # Last file gives end values
        d_last = json.load(open(paths[-1]))
        w0_last = d_last["psc_w0_vel"]["values"][-1] if "psc_w0_vel" in d_last else float("nan")
        w1_last = d_last["psc_w1_ang"]["values"][-1] if "psc_w1_ang" in d_last else float("nan")
        b_last = d_last["psc_bias"]["values"][-1] if "psc_bias" in d_last else float("nan")
        print(f"  {s:>5d}  {w0_first:>10.4f} {w0_last:>10.4f}  {w1_first:>10.4f} {w1_last:>10.4f}  {b_first:>10.4f} {b_last:>10.4f}")

# Plot
print()
print("§4 Generating B/D/E comparison plot...")
try:
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    out_dir = "experiments/_E_analysis"
    os.makedirs(out_dir, exist_ok=True)
    color = {"B": "#bb6666", "D": "#22aa44", "E": "#3377cc"}
    label = {"B": "B: MLP+sparse", "D": "D: PSC+sparse (learnable w)", "E": "E: PSC+sparse (FIXED w)"}

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    for g in "BDE":
        all_curves = []
        ref_steps = None
        for s in SEEDS:
            steps, vals = stitch(g, s)
            if ref_steps is None or len(steps) > len(ref_steps):
                ref_steps = np.array(steps)
            all_curves.append(np.interp(ref_steps if ref_steps is not None else steps, steps, vals,
                                         left=vals[0] if vals else 0, right=vals[-1] if vals else 0))
        arr = np.array(all_curves)
        med = np.median(arr, axis=0)
        q1 = np.percentile(arr, 25, axis=0)
        q3 = np.percentile(arr, 75, axis=0)
        ax.plot(ref_steps, med, label=label[g], color=color[g], lw=2)
        ax.fill_between(ref_steps, q1, q3, color=color[g], alpha=0.2)
    ax.set_title("PSC vs PBRS empirical: B (no PSC) / D (learnable PSC) / E (fixed PSC)")
    ax.set_xlabel("env steps")
    ax.set_ylabel("mean reward")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/B_D_E_comparison.png", dpi=120)
    plt.close(fig)
    print(f"    [+] {out_dir}/B_D_E_comparison.png")

except Exception as e:
    print(f"    [!] plot failed: {e}")

print()
print("=" * 90)
print("Done.")
