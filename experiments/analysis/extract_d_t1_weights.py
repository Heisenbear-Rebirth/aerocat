"""Extract D-T1 converged PSC weights (5-seed mean) for E1 cross-task transfer."""
import glob, json, os
import numpy as np

SEEDS = [42, 123, 456, 789, 1024]
KEYS = ["psc_w0_vel", "psc_w1_ang", "psc_w2_tilt", "psc_w3_int", "psc_w4_sat", "psc_bias"]


def stitch_last(seed, key):
    pairs = []
    for p in sorted(glob.glob(
            f"experiments/ablation_D_sparse_psc/seed_{seed}/*/*/metrics.json")):
        with open(p) as f:
            d = json.load(f)
        if key in d:
            for st, v in zip(d[key]["steps"], d[key]["values"]):
                pairs.append((st, v))
    if not pairs:
        return None
    pairs.sort(key=lambda x: x[0])
    return pairs[-1][1]  # value at largest step


per_seed = {k: [] for k in KEYS}
for s in SEEDS:
    for k in KEYS:
        v = stitch_last(s, k)
        if v is not None:
            per_seed[k].append(v)
        print(f"seed {s} {k}: {v}")

print("\n=== D-T1 converged PSC weights (5-seed mean) ===")
mean_w = []
for k in KEYS:
    arr = np.array(per_seed[k])
    m = float(arr.mean())
    sd = float(arr.std(ddof=1))
    print(f"{k}: mean={m:.4f}  sd={sd:.4f}  (n={len(arr)})")
    mean_w.append(m)

print("\n--psc-init-w "
      f"{mean_w[0]:.4f} {mean_w[1]:.4f} {mean_w[2]:.4f} {mean_w[3]:.4f} {mean_w[4]:.4f} "
      f"--psc-init-b {mean_w[5]:.4f}")
