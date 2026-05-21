"""
B1+ Cross-validation: RL-spontaneous PSC weight drift vs E2 leave-one-out SR drop.

Hypothesis: in D-T1 training, RL drifts w_i AWAY from init only for bases that
actually carry signal. Bases RL drifts toward 0 (or reverses sign of) are
implicitly being "removed" by RL. The empirical E2 leave-one-out (forced w_i=0)
should therefore agree: bases RL self-suppressed are bases E2 finds redundant
or even harmful to keep.

Per-(basis, seed) we compute:
  drift_i_s = |w_i_final - w_i_init|         (RL's spontaneous reshape strength)
  sr_change_i_s = SR_noPhi_i[seed=s] - SR_full_D[seed=s]
                  (positive = removing basis HELPS, negative = removing HURTS)

Two analyses:
  Level A (per-basis, n=5 bases): average drift_i and sr_change_i over 5 seeds,
                                  see if drift magnitude correlates with leave-one-out impact.
  Level B (per-(basis, seed), n=25): all pairs.

Init values: [45, 2, 2, 0.5, 1] for w; 20 for b.
"""
import glob, json
import numpy as np

SEEDS = [42, 123, 456, 789, 1024]
W_KEYS = ["psc_w0_vel", "psc_w1_ang", "psc_w2_tilt", "psc_w3_int", "psc_w4_sat"]
W_INIT = np.array([45.0, 2.0, 2.0, 0.5, 1.0])
BASIS_NAME = {
    0: "phi0 vel_err",
    1: "phi1 omega",
    2: "phi2 tilt",
    3: "phi3 PID_integral",
    4: "phi4 saturation",
}

OUT_DIR = "experiments/_b1plus_drift_vs_e2"
import os
os.makedirs(OUT_DIR, exist_ok=True)


def stitch_last(dir_pat, seed, key):
    pairs = []
    for p in sorted(glob.glob(dir_pat.format(s=seed))):
        with open(p) as f:
            d = json.load(f)
        if key in d:
            for st, v in zip(d[key]["steps"], d[key]["values"]):
                pairs.append((st, v))
    if not pairs:
        return None
    pairs.sort(key=lambda x: x[0])
    return pairs[-1][1]


def stitch_plateau(dir_pat, seed, key="success_rate", frac=0.1):
    pairs = []
    for p in sorted(glob.glob(dir_pat.format(s=seed))):
        with open(p) as f:
            d = json.load(f)
        if key in d:
            for st, v in zip(d[key]["steps"], d[key]["values"]):
                pairs.append((st, v))
    if not pairs:
        return None
    pairs.sort(key=lambda x: x[0])
    vals = [p[1] for p in pairs]
    n = max(1, int(len(vals) * frac))
    return float(np.mean(vals[-n:]))


def main():
    # ---- (1) D-T1 final w_i per seed ----
    print("[*] extracting D-T1 final weights per seed...")
    w_final = np.zeros((5, 5))  # [seed, basis]
    for is_, s in enumerate(SEEDS):
        for ib, k in enumerate(W_KEYS):
            v = stitch_last(
                "experiments/ablation_D_sparse_psc/seed_{s}/*/*/metrics.json",
                s, k)
            if v is None:
                raise RuntimeError(f"missing {k} for seed {s}")
            w_final[is_, ib] = v
    drift_signed = w_final - W_INIT[None, :]            # [seed, basis]
    drift_abs = np.abs(drift_signed)
    print(f"    w_final shape: {w_final.shape}")

    # ---- (2) D-T1 baseline SR plateau per seed ----
    print("[*] extracting D-T1 baseline SR plateau per seed...")
    sr_full = np.array([
        stitch_plateau("experiments/ablation_D_sparse_psc/seed_{s}/*/*/metrics.json", s)
        for s in SEEDS
    ])
    print(f"    D baseline SR per seed: {sr_full}")

    # ---- (3) E2 noPhi_i SR plateau per seed ----
    print("[*] extracting E2 noPhi_i SR per seed...")
    sr_no = np.zeros((5, 5))  # [seed, basis]
    for is_, s in enumerate(SEEDS):
        for ib in range(5):
            v = stitch_plateau(
                f"experiments/ablation_D_sparse_psc_noPhi{ib}/seed_{{s}}/*/*/metrics.json", s)
            if v is None:
                raise RuntimeError(f"missing noPhi{ib} seed {s}")
            sr_no[is_, ib] = v
    sr_change = sr_no - sr_full[:, None]              # [seed, basis]
    print(f"    sr_change shape: {sr_change.shape}")

    # ---- (4) Level A: per-basis means ----
    print()
    print("=" * 78)
    print("Level A: per-basis (n=5 bases, 5-seed averaged)")
    print("=" * 78)
    print(f"{'basis':22s}  drift_signed(mean)  |drift|(mean)  SR_change(mean)  SR_change(SD)")
    drift_mean = drift_signed.mean(axis=0)
    drift_abs_mean = drift_abs.mean(axis=0)
    sr_change_mean = sr_change.mean(axis=0)
    sr_change_sd = sr_change.std(axis=0, ddof=1)
    for ib in range(5):
        print(f"  {BASIS_NAME[ib]:20s}  {drift_mean[ib]:+10.4f}        {drift_abs_mean[ib]:10.4f}    "
              f"{sr_change_mean[ib]:+.4f}        {sr_change_sd[ib]:.4f}")

    # Pearson corr at basis level (n=5)
    from scipy.stats import pearsonr, spearmanr
    rho_a_pearson, p_a_pearson = pearsonr(drift_abs_mean, sr_change_mean)
    rho_a_spearman, p_a_spearman = spearmanr(drift_abs_mean, sr_change_mean)
    print(f"\n  basis-level Pearson  corr(|drift|, SR_change) = {rho_a_pearson:+.3f}  p={p_a_pearson:.3f}")
    print(f"  basis-level Spearman corr(|drift|, SR_change) = {rho_a_spearman:+.3f}  p={p_a_spearman:.3f}")
    print("  (n=5 bases — interpret with caution)")

    # ---- (5) Level B: per-(basis, seed) ----
    print()
    print("=" * 78)
    print("Level B: per-(basis, seed) (n=25 pairs)")
    print("=" * 78)
    drift_flat = drift_abs.flatten()
    sr_change_flat = sr_change.flatten()
    rho_b_pearson, p_b_pearson = pearsonr(drift_flat, sr_change_flat)
    rho_b_spearman, p_b_spearman = spearmanr(drift_flat, sr_change_flat)
    print(f"  Pearson  corr(|drift|, SR_change) = {rho_b_pearson:+.3f}  p={p_b_pearson:.3f}")
    print(f"  Spearman corr(|drift|, SR_change) = {rho_b_spearman:+.3f}  p={p_b_spearman:.3f}")

    # ---- (6) Sign-of-drift analysis: did RL push w_i toward 0? ----
    print()
    print("=" * 78)
    print("Did RL push w_i toward zero? Signed drift (final - init)")
    print("=" * 78)
    for ib in range(5):
        init_v = W_INIT[ib]
        signed = drift_signed[:, ib]
        # if init > 0, "toward 0" means signed < 0
        toward_zero = "yes" if (init_v > 0 and signed.mean() < 0) or (init_v < 0 and signed.mean() > 0) else "no"
        crossed = "yes" if np.any(np.sign(w_final[:, ib]) != np.sign(init_v)) else "no"
        e2_verdict = ("HELPFUL to remove" if sr_change_mean[ib] > 0 and abs(sr_change_mean[ib]/sr_change_sd[ib]) > 1 else
                      "harmful to remove" if sr_change_mean[ib] < 0 and abs(sr_change_mean[ib]/sr_change_sd[ib]) > 1 else
                      "n.s. drop")
        print(f"  {BASIS_NAME[ib]:20s}  init={init_v:+.2f}  final_mean={w_final.mean(axis=0)[ib]:+.3f}  "
              f"toward_0={toward_zero}  crossed_sign={crossed}  "
              f"E2: SR_change={sr_change_mean[ib]:+.4f} -> {e2_verdict}")

    # Write markdown table
    lines = []
    lines.append("# B1+ Cross-Validation: RL spontaneous weight drift vs E2 leave-one-out\n")
    lines.append("Hypothesis: RL drift magnitude per basis should agree with E2 leave-one-out impact.\n")
    lines.append("\n## Per-basis means (n=5 bases, 5-seed averaged)\n")
    lines.append("| basis | init w | final w mean | drift mean | |drift| mean | E2 SR_change mean | E2 SR_change SD |")
    lines.append("|:--|:--:|:--:|:--:|:--:|:--:|:--:|")
    for ib in range(5):
        lines.append(
            f"| {BASIS_NAME[ib]} | {W_INIT[ib]:+.2f} | {w_final.mean(axis=0)[ib]:+.3f} | "
            f"{drift_mean[ib]:+.3f} | {drift_abs_mean[ib]:.3f} | "
            f"{sr_change_mean[ib]:+.4f} | {sr_change_sd[ib]:.4f} |"
        )
    lines.append(f"\n**Correlations (n=5 bases)**: Pearson |drift| vs SR_change = {rho_a_pearson:+.3f} (p={p_a_pearson:.3f}); "
                 f"Spearman = {rho_a_spearman:+.3f} (p={p_a_spearman:.3f}).")
    lines.append(f"\n**Correlations (n=25 basis-seed pairs)**: Pearson = {rho_b_pearson:+.3f} (p={p_b_pearson:.3f}); "
                 f"Spearman = {rho_b_spearman:+.3f} (p={p_b_spearman:.3f}).")
    lines.append("\n## Sign-of-drift commentary\n")
    for ib in range(5):
        init_v = W_INIT[ib]
        crossed = np.any(np.sign(w_final[:, ib]) != np.sign(init_v))
        if abs(drift_abs_mean[ib]) > 1.0:
            note = f"**large |drift| = {drift_abs_mean[ib]:.2f}** — RL actively reshaped this weight"
        else:
            note = f"|drift| = {drift_abs_mean[ib]:.2f} — RL barely moved this weight"
        cross_note = " (sign crossed across seeds)" if crossed else ""
        lines.append(f"- **{BASIS_NAME[ib]}**: init {init_v:+.2f} → final {w_final.mean(axis=0)[ib]:+.3f}; "
                     f"{note}{cross_note}. E2 verdict: ΔSR = {sr_change_mean[ib]:+.4f} ± {sr_change_sd[ib]:.4f}")

    with open(f"{OUT_DIR}/b1plus_table.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n[+] wrote {OUT_DIR}/b1plus_table.md")


if __name__ == "__main__":
    main()
