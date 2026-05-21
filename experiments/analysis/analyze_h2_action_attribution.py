"""
H2 — Action spectrum attribution (per-channel FFT + state-action lagged correlation).

C2 already showed sparse-trained policies (B/D/E) emit ~33-39% smaller commands than
dense-trained (A/C/F), and conservatism is attributed to reward composition, not PSC.
H2 asks a finer-grained question:
  Q1: Is the smaller magnitude due to LOWER FREQUENCY content (smooth low-bandwidth)
      or LOWER AMPLITUDE at all frequencies?
  Q2: Does D's policy respond to v_err with a different time-lag than A's?
       (lagged correlation: action[t-k] vs v_err[t], find peak lag)

Per-channel FFT (action shape (T=500, E=256, 4)):
  - Power spectrum per env, mean across envs
  - Aggregate energy in 3 bands assuming 50 Hz control rate:
      low:  0.5–5 Hz   (slow trim / steady-state)
      mid:  5–15 Hz    (active correction)
      high: 15–25 Hz   (Nyquist limit; rate-saturation regime)
  - Ratio = E_high / E_total per channel

Lagged correlation:
  - For each env, compute Pearson(action_norm[t-k], v_err[t]) for k=0..20
  - action_norm = |action|_2 normalized; v_err = velocity-error magnitude
  - Find lag k* maximizing |corr|; aggregate per (group, lam) median k*

Paired contrasts at lambda=1.0:
  D vs A (PSC+sparse vs MLP+dense): mainline cross-reward contrast
  D vs B (PSC+sparse vs MLP+sparse): isolate PSC effect within sparse

Outputs:
  experiments/_h2_action_attribution/
    h2_table.md
    h2_action_spectrum.{pdf,png}        spectra for A vs D at lambda=1.0
    h2_band_energy_bars.{pdf,png}       3-band energy bars per group at lambda=1.0
    h2_per_cell.json
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
GROUPS = ["A", "B", "C", "D", "E", "F"]
SEEDS = [42, 123, 456, 789, 1024]
LAMBDAS = [0.0, 0.3, 0.5, 0.7, 1.0]
ACTION_NAMES = ["roll_rate", "pitch_rate", "yaw_rate", "thrust"]

CTRL_HZ = 50.0  # control rate (T1)
BANDS = {"low": (0.5, 5.0), "mid": (5.0, 15.0), "high": (15.0, 25.0)}
MAX_LAG = 20

TRAJ_DIR = "experiments/_c2d1/T1_trajectories"
OUT = "experiments/_h2_action_attribution"


def load_cell(g, s, lam):
    fn = f"{TRAJ_DIR}/traj_{g}_{s}_lam{lam}.npz"
    if not os.path.exists(fn): return None
    z = np.load(fn, allow_pickle=True)
    return dict(action=z["action"], saturation=z["saturation"],
                tilt=z["tilt"], v_err=z["v_err"], active=z["active"])


def action_fft_bands(action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (freqs, mean_psd_per_channel (4, F), band_energies (4, 3))."""
    T, E, C = action.shape
    # PSD per env per channel (rfft of mean-subtracted)
    a_centered = action - action.mean(axis=0, keepdims=True)
    # FFT along T axis, average abs^2 over envs
    Y = np.fft.rfft(a_centered, axis=0)         # (F, E, C)
    psd = np.abs(Y) ** 2 / T
    psd_mean = psd.mean(axis=1)                  # (F, C) mean over envs
    freqs = np.fft.rfftfreq(T, d=1.0 / CTRL_HZ)  # (F,)

    band_energies = np.zeros((C, 3), dtype=np.float64)
    for ci in range(C):
        total = psd_mean[:, ci].sum()
        if total <= 0:
            band_energies[ci, :] = np.nan; continue
        for bi, (key, (lo, hi)) in enumerate(BANDS.items()):
            mask = (freqs >= lo) & (freqs <= hi)
            band_energies[ci, bi] = psd_mean[mask, ci].sum() / total
    return freqs, psd_mean.T, band_energies  # (4, F) and (4, 3)


def lagged_peak_corr(action: np.ndarray, v_err: np.ndarray, max_lag: int = MAX_LAG):
    """For each env, find lag k in [0, max_lag] maximizing |corr(|a|_2[t-k], v_err[t])|.
    Returns (median k*, median |peak corr|) across envs."""
    T, E, C = action.shape
    a_mag = np.linalg.norm(action, axis=-1)  # (T, E)
    # center
    a_c = a_mag - a_mag.mean(axis=0, keepdims=True)
    v_c = v_err - v_err.mean(axis=0, keepdims=True)
    a_std = a_c.std(axis=0)
    v_std = v_c.std(axis=0)
    peak_lags = []
    peak_corrs = []
    for e in range(E):
        if a_std[e] < 1e-8 or v_std[e] < 1e-8:
            continue
        cs = []
        for k in range(0, max_lag + 1):
            if T - k < 5: break
            a_slice = a_c[:T - k, e]
            v_slice = v_c[k:, e]
            num = (a_slice * v_slice).sum()
            den = np.sqrt((a_slice ** 2).sum() * (v_slice ** 2).sum())
            cs.append(num / den if den > 0 else 0.0)
        cs = np.array(cs)
        if cs.size == 0: continue
        k_star = int(np.argmax(np.abs(cs)))
        peak_lags.append(k_star)
        peak_corrs.append(cs[k_star])
    if not peak_lags:
        return np.nan, np.nan
    return float(np.median(peak_lags)), float(np.median(peak_corrs))


def process_cell(data: dict) -> dict:
    action = data["action"].astype(np.float32)
    v_err = data["v_err"].astype(np.float32)
    freqs, psd_per_ch, band_e = action_fft_bands(action)
    lag, lag_corr = lagged_peak_corr(action, v_err)
    return dict(
        freqs=freqs.tolist(),
        psd_per_ch=psd_per_ch.tolist(),
        band_energies_per_ch=band_e.tolist(),       # (4, 3): rows = channels, cols = low/mid/high
        peak_lag_median=lag,
        peak_corr_median=lag_corr,
        action_l2_mean=float(np.linalg.norm(action, axis=-1).mean()),
    )


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
                cell = load_cell(g, s, lam)
                if cell is None:
                    raw[(g, s, lam)] = None
                else:
                    raw[(g, s, lam)] = process_cell(cell)

    # JSON dump
    dump = {f"{g}_{s}_lam{lam}": v for (g, s, lam), v in raw.items() if v is not None}
    with open(f"{OUT}/h2_per_cell.json", "w") as f:
        json.dump(dump, f, indent=2)
    print(f"[+] wrote {OUT}/h2_per_cell.json ({len(dump)} cells)")

    # ============================================================
    # Table
    # ============================================================
    lines = []
    lines.append("# H2 — Action spectrum & state-action lag attribution\n")
    lines.append(f"Assuming control rate {CTRL_HZ:.0f} Hz; bands: low={BANDS['low']}, mid={BANDS['mid']}, high={BANDS['high']} Hz.\n")
    lines.append(f"Per-channel: roll_rate, pitch_rate, yaw_rate, thrust.\n")

    lines.append("\n## (1) High-band fraction E_high/E_total per channel at λ=1.0 (mean ± SD over 5 seeds)\n")
    lines.append("Higher value = more high-frequency content in the command. Lower = smoother control.\n")
    lines.append("| Group | roll | pitch | yaw | thrust |")
    lines.append("|:---:|:---:|:---:|:---:|:---:|")
    for g in GROUPS:
        row = [g]
        for ci in range(4):
            vals = []
            for s in SEEDS:
                cell = raw.get((g, s, 1.0))
                if cell is None: continue
                be = np.array(cell["band_energies_per_ch"])
                v = be[ci, 2]  # high band
                if not np.isnan(v): vals.append(v)
            if len(vals) < 2:
                row.append("—")
            else:
                row.append(f"{np.mean(vals):.3f}±{np.std(vals, ddof=1):.3f}")
        lines.append("| " + " | ".join(row) + " |")

    lines.append("\n## (2) Paired Δ(high-band fraction) per channel at λ=1.0\n")
    lines.append("Negative ⇒ sparse policy has SMOOTHER (lower high-freq) commands.\n")
    lines.append("| Contrast | roll | pitch | yaw | thrust |")
    lines.append("|:---:|:---:|:---:|:---:|:---:|")
    contrasts = [("D − A", "D", "A"), ("D − B", "D", "B"), ("D − E", "D", "E"),
                 ("B − A", "B", "A"), ("C − A", "C", "A"), ("F − A", "F", "A")]
    for label, ga, gb in contrasts:
        row = [label]
        for ci in range(4):
            diffs = []
            for s in SEEDS:
                ra = raw.get((ga, s, 1.0)); rb = raw.get((gb, s, 1.0))
                if ra is None or rb is None: continue
                va = np.array(ra["band_energies_per_ch"])[ci, 2]
                vb = np.array(rb["band_energies_per_ch"])[ci, 2]
                if np.isnan(va) or np.isnan(vb): continue
                diffs.append(va - vb)
            m, t, d, n = paired_t(diffs)
            if np.isnan(m):
                row.append("—")
            else:
                sig = "**" if abs(t) > 2.776 else ("·" if abs(t) > 2.132 else "")
                row.append(f"{m:+.3f}{sig} (t={t:+.1f})")
        lines.append("| " + " | ".join(row) + " |")

    lines.append("\n** = p<0.05 (|t|>2.776, df=4); · = p<0.10 (|t|>2.132).\n")

    # ---- low-band fraction (steady-state share) ----
    lines.append("\n## (3) Low-band fraction E_low/E_total per channel at λ=1.0\n")
    lines.append("Higher = more steady-state/trim content; lower = more transient/corrective.\n")
    lines.append("| Group | roll | pitch | yaw | thrust |")
    lines.append("|:---:|:---:|:---:|:---:|:---:|")
    for g in GROUPS:
        row = [g]
        for ci in range(4):
            vals = []
            for s in SEEDS:
                cell = raw.get((g, s, 1.0))
                if cell is None: continue
                be = np.array(cell["band_energies_per_ch"])
                v = be[ci, 0]
                if not np.isnan(v): vals.append(v)
            if len(vals) < 2:
                row.append("—")
            else:
                row.append(f"{np.mean(vals):.3f}±{np.std(vals, ddof=1):.3f}")
        lines.append("| " + " | ".join(row) + " |")

    # ---- Lag analysis ----
    lines.append("\n## (4) State-action peak-lag-correlation: median lag (steps) and corr at λ=1.0\n")
    lines.append("Lagged Pearson(|action|_2[t-k], v_err[t]); peak |corr| over k∈[0,20].\n")
    lines.append("Higher lag ⇒ action responds to v_err with delay (anticipatory? or sluggish?).\n")
    lines.append("| Group | median peak lag (steps) | median |peak corr| | n |")
    lines.append("|:---:|:---:|:---:|:---:|")
    for g in GROUPS:
        lags = []; corrs = []
        for s in SEEDS:
            cell = raw.get((g, s, 1.0))
            if cell is None: continue
            l = cell["peak_lag_median"]
            c = cell["peak_corr_median"]
            if not np.isnan(l): lags.append(l)
            if not np.isnan(c): corrs.append(c)
        if len(lags) < 2:
            lines.append(f"| {g} | — | — | {len(lags)} |")
        else:
            lines.append(f"| {g} | {np.mean(lags):.1f}±{np.std(lags, ddof=1):.1f} | "
                         f"{np.mean(corrs):+.3f}±{np.std(corrs, ddof=1):.3f} | {len(lags)} |")

    with open(f"{OUT}/h2_table.md", "w") as f:
        f.write("\n".join(lines))
    print(f"[+] wrote {OUT}/h2_table.md")

    # ============================================================
    # Figures
    # ============================================================
    # Fig 1: action PSD overlay at lambda=1.0 (averaged across seeds), 4 channels in subplots
    # Cells may have different T (truncated when all envs done), so interpolate to common grid.
    F_COMMON = np.linspace(0, CTRL_HZ / 2, 128)
    fig, axes = plt.subplots(2, 2, figsize=(DBL_WIDTH, ROW_H * 2.0))
    axes = axes.flatten()
    for ci in range(4):
        ax = axes[ci]
        for g in ["A", "D", "F"]:
            psds_interp = []
            for s in SEEDS:
                cell = raw.get((g, s, 1.0))
                if cell is None: continue
                psd = np.array(cell["psd_per_ch"])[ci]
                f = np.array(cell["freqs"])
                psd_i = np.interp(F_COMMON, f, psd, left=psd[0], right=psd[-1])
                psds_interp.append(psd_i)
            if not psds_interp: continue
            psd_mean = np.median(np.stack(psds_interp), axis=0)
            ax.plot(F_COMMON, psd_mean, color=COLORS[g], label=LABELS[g], lw=1.3)
        ax.set_yscale("log")
        ax.set_xlim(0, CTRL_HZ / 2)
        ax.set_xlabel("Hz")
        ax.set_ylabel("PSD (log)")
        ax.set_title(ACTION_NAMES[ci], fontsize=9)
        ax.legend(loc="best", framealpha=0.85, edgecolor='none', fontsize=7)
    plt.tight_layout(pad=0.5)
    fig.savefig(f"{OUT}/h2_action_spectrum.pdf"); fig.savefig(f"{OUT}/h2_action_spectrum.png", dpi=160)
    plt.close(fig)
    print(f"[+] wrote {OUT}/h2_action_spectrum.pdf")

    # Fig 2: high-band fraction bars at lambda=1.0 across groups
    fig, ax = plt.subplots(1, 1, figsize=(DBL_WIDTH, ROW_H))
    width = 0.13
    x = np.arange(4)
    for i, g in enumerate(GROUPS):
        means = []
        sds = []
        for ci in range(4):
            vals = []
            for s in SEEDS:
                cell = raw.get((g, s, 1.0))
                if cell is None: continue
                be = np.array(cell["band_energies_per_ch"])
                v = be[ci, 2]
                if not np.isnan(v): vals.append(v)
            if len(vals) < 2: means.append(np.nan); sds.append(0); continue
            means.append(np.mean(vals)); sds.append(np.std(vals, ddof=1))
        ax.bar(x + i * width - 2.5 * width, means, width, color=COLORS[g], label=LABELS[g],
               yerr=sds, capsize=2, edgecolor='black', linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(ACTION_NAMES)
    ax.set_ylabel("$E_{\\mathrm{high}}/E_{\\mathrm{total}}$ (15–25 Hz)")
    ax.set_title("Action high-frequency content at OOD λ=1.0", fontsize=10)
    ax.legend(loc="upper left", framealpha=0.85, edgecolor='none', fontsize=7, ncol=2)
    plt.tight_layout(pad=0.5)
    fig.savefig(f"{OUT}/h2_band_energy_bars.pdf"); fig.savefig(f"{OUT}/h2_band_energy_bars.png", dpi=160)
    plt.close(fig)
    print(f"[+] wrote {OUT}/h2_band_energy_bars.pdf")

    # ============================================================
    # Verdict
    # ============================================================
    print()
    print("=" * 72)
    print("H2 verdict — does sparse policy have smoother (lower high-freq) commands?")
    print("=" * 72)
    print(f"\nHigh-band fraction (E_high/E_total) at λ=1.0:")
    for g in GROUPS:
        per_ch = []
        for ci in range(4):
            vals = []
            for s in SEEDS:
                cell = raw.get((g, s, 1.0))
                if cell is None: continue
                be = np.array(cell["band_energies_per_ch"])
                v = be[ci, 2]
                if not np.isnan(v): vals.append(v)
            per_ch.append(np.mean(vals) if len(vals) >= 2 else np.nan)
        print(f"  {g}: roll={per_ch[0]:.3f}  pitch={per_ch[1]:.3f}  yaw={per_ch[2]:.3f}  thrust={per_ch[3]:.3f}")

    print(f"\nPaired Δ(high-fraction) at λ=1.0:")
    for label, ga, gb in contrasts:
        per_ch = []
        for ci in range(4):
            diffs = []
            for s in SEEDS:
                ra = raw.get((ga, s, 1.0)); rb = raw.get((gb, s, 1.0))
                if ra is None or rb is None: continue
                va = np.array(ra["band_energies_per_ch"])[ci, 2]
                vb = np.array(rb["band_energies_per_ch"])[ci, 2]
                if np.isnan(va) or np.isnan(vb): continue
                diffs.append(va - vb)
            m, t, _, n = paired_t(diffs)
            per_ch.append((m, t, n))
        print(f"  {label}: " + "  ".join([f"{nm}:{m:+.3f}(t={t:+.1f})" for nm, (m, t, n) in zip(ACTION_NAMES, per_ch)]))

    print(f"\nState-action peak lag at λ=1.0 (median across seeds):")
    for g in GROUPS:
        lags = []; corrs = []
        for s in SEEDS:
            cell = raw.get((g, s, 1.0))
            if cell is None: continue
            l = cell["peak_lag_median"]; c = cell["peak_corr_median"]
            if not np.isnan(l): lags.append(l)
            if not np.isnan(c): corrs.append(c)
        if len(lags) >= 2:
            print(f"  {g}: lag={np.mean(lags):.1f}±{np.std(lags, ddof=1):.1f} steps   |peak corr|={np.mean(corrs):+.3f}")
    print()


if __name__ == "__main__":
    main()
