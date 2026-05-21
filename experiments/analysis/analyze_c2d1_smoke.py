"""
C2+D1 smoke-test analysis — compare policy behavior of D (PSC+sparse) vs A (MLP+dense)
in OOD λ=1.0 trajectories. The goal of *this* script is to answer one question:

  Is the per-step action / saturation / commanded-rate distribution visibly different
  between D and A in the OOD λ=1.0 regime, enough to merit a full re-run?

We compute (per cell):
  C2 metrics:
    - |action|_2  distribution (active env-step weighted)
    - per-dim action distribution (roll_rate, pitch_rate, yaw_rate, thrust)
    - saturation magnitude distribution
    - saturation-triggering rate: P(saturation > 0.5)

  D1 metrics:
    - jerk = ||a[t] - a[t-1]||_2 over active env-step pairs
    - per-dim jerk
    - rough high-frequency-energy proxy: per-env action variance over its trajectory

Outputs a verdict to stdout.
"""
import glob
import os
import numpy as np

TRAJ_DIR = "experiments/_c2d1/smoke_trajectories"


def load_cell(path):
    z = np.load(path)
    return {
        "action":     z["action"],     # (T, E, 4)
        "saturation": z["saturation"], # (T, E)
        "tilt":       z["tilt"],       # (T, E)
        "v_err":      z["v_err"],      # (T, E)
        "active":     z["active"],     # (T, E)
    }


def flatten_active(arr, active):
    """Return flat 1-D array of values where active==1.0. arr can be (T,E) or (T,E,K)."""
    if arr.ndim == 2:
        mask = (active > 0.5).reshape(-1)
        return arr.reshape(-1)[mask]
    else:
        T, E, K = arr.shape
        mask = (active > 0.5).reshape(-1)
        return arr.reshape(T * E, K)[mask]


def jerk_active(action, active):
    """Per-step jerk = ||a[t]-a[t-1]||_2. Only emit where active[t]=active[t-1]=1."""
    da = action[1:] - action[:-1]                    # (T-1, E, 4)
    jerk = np.linalg.norm(da, axis=-1)               # (T-1, E)
    mask = (active[1:] > 0.5) & (active[:-1] > 0.5)
    return jerk[mask]


def per_env_action_std(action, active):
    """Per-env std of action over its time series (averaged over active steps)."""
    T, E, K = action.shape
    out = np.zeros(E)
    for e in range(E):
        m = active[:, e] > 0.5
        if m.sum() < 2:
            out[e] = np.nan
        else:
            out[e] = np.linalg.norm(action[m, e].std(axis=0))
    return out[~np.isnan(out)]


def summarize(name, x):
    if len(x) == 0:
        return f"  {name:24s}: empty"
    return (f"  {name:24s}: n={len(x):>7d}  "
            f"mean={x.mean():.4g}  sd={x.std():.4g}  "
            f"p50={np.percentile(x, 50):.4g}  p95={np.percentile(x, 95):.4g}")


def main():
    paths = sorted(glob.glob(f"{TRAJ_DIR}/traj_*.npz"))
    if not paths:
        print(f"[!] no trajectory files in {TRAJ_DIR}")
        return

    cells = {}
    for p in paths:
        name = os.path.basename(p).replace(".npz", "")
        parts = name.split("_")
        # traj_<G>_<seed>_lam<L>
        g = parts[1]
        seed = parts[2]
        lam = parts[3]
        cells[(g, seed, lam)] = load_cell(p)

    print(f"[+] loaded {len(cells)} cells")
    for k in cells:
        print(f"    {k}: action shape = {cells[k]['action'].shape}")
    print()

    # Aggregate per-group (only A and D in smoke)
    by_group = {}
    for (g, s, lam), c in cells.items():
        by_group.setdefault(g, []).append(c)

    print("=" * 72)
    print("Per-group active-step distributions")
    print("=" * 72)

    metrics = {}
    for g, cell_list in by_group.items():
        print(f"\n--- Group {g} ---")
        action_all = np.concatenate([flatten_active(c["action"], c["active"]) for c in cell_list], axis=0)
        sat_all    = np.concatenate([flatten_active(c["saturation"], c["active"]) for c in cell_list], axis=0)
        v_err_all  = np.concatenate([flatten_active(c["v_err"], c["active"]) for c in cell_list], axis=0)
        tilt_all   = np.concatenate([flatten_active(c["tilt"], c["active"]) for c in cell_list], axis=0)
        jerk_all   = np.concatenate([jerk_active(c["action"], c["active"]) for c in cell_list], axis=0)
        env_std_all = np.concatenate([per_env_action_std(c["action"], c["active"]) for c in cell_list], axis=0)

        anorm = np.linalg.norm(action_all, axis=-1)
        sat_trig = float((sat_all > 0.5).mean())

        print(summarize("|action|_2", anorm))
        print(summarize("roll_rate (a[0])", action_all[:, 0]))
        print(summarize("pitch_rate (a[1])", action_all[:, 1]))
        print(summarize("yaw_rate (a[2])", action_all[:, 2]))
        print(summarize("thrust (a[3])", action_all[:, 3]))
        print(summarize("saturation", sat_all))
        print(f"  {'sat triggering rate':24s}: P(sat>0.5) = {sat_trig:.4f}")
        print(summarize("v_err (m/s)", v_err_all))
        print(summarize("tilt (rad²)", tilt_all))
        print(summarize("jerk (step-to-step ΔA)", jerk_all))
        print(summarize("per-env action std", env_std_all))

        metrics[g] = {
            "anorm_mean": anorm.mean(),
            "anorm_p95": np.percentile(anorm, 95),
            "sat_mean": sat_all.mean(),
            "sat_p95": np.percentile(sat_all, 95),
            "sat_trig_rate": sat_trig,
            "v_err_mean": v_err_all.mean(),
            "tilt_mean": tilt_all.mean(),
            "tilt_p95": np.percentile(tilt_all, 95),
            "jerk_mean": jerk_all.mean(),
            "jerk_p95": np.percentile(jerk_all, 95),
            "env_std_mean": env_std_all.mean(),
        }

    # Compare D vs A
    print()
    print("=" * 72)
    print("D vs A comparison (D - A; positive = D higher)")
    print("=" * 72)
    if "A" in metrics and "D" in metrics:
        for k in metrics["A"]:
            da = metrics["D"][k] - metrics["A"][k]
            a_val = metrics["A"][k]
            ratio = (da / a_val * 100) if a_val != 0 else float("nan")
            print(f"  {k:22s}: D={metrics['D'][k]:.4f}  A={a_val:.4f}  Δ={da:+.4f}  ({ratio:+.1f}%)")

    print()
    print("Smoke-test verdict heuristic:")
    print("  → Strong signal if |relative Δ| > 30% on ≥2 of: anorm_mean/sat_trig_rate/jerk_mean/tilt_p95")
    print("  → Weak signal if all |relative Δ| < 10%  →  consider aborting full re-run")


if __name__ == "__main__":
    main()
