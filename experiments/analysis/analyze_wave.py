"""
Analyze a single-seed wave of 4-group ablation experiments.

Generates:
  - per-group summary tables (initial/mid/final mean_reward, vf_loss, td_*_std, psc_w trajectory)
  - 2x2 group-comparison plots: mean_reward, vf_loss, advantage_std, td_res_std
  - PSC weight trajectory plot for groups C/D
  - cross-group bar chart at final step

Usage:
    python experiments/analyze_wave.py [--wave 2]
"""

import argparse
import json
import os
import glob
from typing import Dict, List, Tuple


GROUPS = {
    "A": ("dense", "mlp"),
    "B": ("sparse", "mlp"),
    "C": ("dense", "psc"),
    "D": ("sparse", "psc"),
}

KEY_METRICS = [
    "mean_reward", "vf_loss", "pg_loss", "entropy", "approx_kl", "clip_frac",
    "success_rate", "curriculum_lambda",
    "v_phys_mean", "v_res_mean", "v_phys_ratio",
    "psc_w0_vel", "psc_w1_ang", "psc_w2_tilt", "psc_w3_int", "psc_w4_sat", "psc_bias",
    "td_error_std", "td_phys_std", "td_res_std", "advantage_std",
]


def find_metrics(base: str, group: str, seed: int) -> str:
    reward_tag, critic_tag = GROUPS[group]
    pattern = os.path.join(
        base, f"ablation_{group}_{reward_tag}_{critic_tag}",
        f"seed_{seed}", "*", "*", "metrics.json",
    )
    matches = sorted(glob.glob(pattern))
    if not matches:
        return ""
    # 取最近一次 (按时间戳目录排序)
    return matches[-1]


def load_metrics(path: str) -> Dict[str, Tuple[List[int], List[float]]]:
    if not path or not os.path.exists(path):
        return {}
    with open(path) as f:
        d = json.load(f)
    out = {}
    for k, v in d.items():
        if isinstance(v, dict) and "values" in v:
            out[k] = (v.get("steps", list(range(len(v["values"])))), v["values"])
        elif isinstance(v, list):
            out[k] = (list(range(len(v))), v)
    return out


def summary_row(label: str, m: Dict[str, Tuple[List[int], List[float]]]) -> str:
    def stat(k: str, mode: str = "last") -> str:
        if k not in m:
            return "  ----  "
        steps, vals = m[k]
        if not vals:
            return "  ----  "
        v = vals[-1] if mode == "last" else (vals[0] if mode == "first" else max(vals))
        return f"{v:8.3f}"

    return (
        f"{label:6s}  "
        f"n={len(m.get('mean_reward', ([],[]))[1]):3d}  "
        f"rew={stat('mean_reward')}  vf={stat('vf_loss')}  pg={stat('pg_loss')}  "
        f"sr={stat('success_rate')}  λ={stat('curriculum_lambda')}  "
        f"vphys={stat('v_phys_mean')}  vres={stat('v_res_mean')}  "
        f"td={stat('td_error_std')}  tdr={stat('td_res_std')}  adv={stat('advantage_std')}"
    )


def plot_group_curves(metrics_by_group: Dict[str, Dict], out_dir: str) -> None:
    """生成 2x2 关键指标对比图 + PSC 权重轨迹图。"""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[!] matplotlib 不可用，跳过画图")
        return

    color = {"A": "#888888", "B": "#bb6666", "C": "#3377cc", "D": "#22aa44"}

    # ===== 图 1: 4 个核心指标 =====
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    panels = [
        ("mean_reward", "Mean reward (rollout)", axes[0, 0]),
        ("vf_loss", "Value function loss", axes[0, 1]),
        ("advantage_std", "Advantage std (GAE noise)", axes[1, 0]),
        ("td_res_std", "TD-residual std (C2 evidence)", axes[1, 1]),
    ]
    for key, title, ax in panels:
        for g, m in metrics_by_group.items():
            if key in m:
                steps, vals = m[key]
                if vals:
                    ax.plot(steps, vals, label=f"Group {g}", color=color[g], lw=1.5)
        ax.set_title(title)
        ax.set_xlabel("env steps")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="best")
        if key == "vf_loss":
            ax.set_yscale("log")

    plt.tight_layout()
    fig_path = os.path.join(out_dir, "wave_groups_4panel.png")
    plt.savefig(fig_path, dpi=110)
    plt.close(fig)
    print(f"[+] saved {fig_path}")

    # ===== 图 2: PSC 权重轨迹 (只 C/D) =====
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    psc_labels = [
        ("psc_w0_vel", "w₀ velocity-error"),
        ("psc_w1_ang", "w₁ angular-velocity"),
        ("psc_w2_tilt", "w₂ tilt"),
        ("psc_w3_int", "w₃ pid-integral"),
        ("psc_w4_sat", "w₄ saturation"),
        ("psc_bias", "b (bias)"),
    ]
    for i, (key, title) in enumerate(psc_labels):
        ax = axes[i // 3, i % 3]
        for g in ["C", "D"]:
            m = metrics_by_group.get(g, {})
            if key in m:
                steps, vals = m[key]
                if vals:
                    ax.plot(steps, vals, label=f"Group {g}", color=color[g], lw=1.5)
        ax.set_title(title)
        ax.set_xlabel("env steps")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="best")

    plt.tight_layout()
    fig_path = os.path.join(out_dir, "wave_psc_weights.png")
    plt.savefig(fig_path, dpi=110)
    plt.close(fig)
    print(f"[+] saved {fig_path}")

    # ===== 图 3: 跨组终值条形图 =====
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    bars_keys = ["mean_reward", "success_rate"]
    for ax, key in zip(axes, bars_keys):
        groups = list(metrics_by_group.keys())
        finals = []
        for g in groups:
            m = metrics_by_group[g]
            v = m.get(key, ([], []))[1]
            finals.append(v[-1] if v else 0.0)
        bars = ax.bar(groups, finals, color=[color[g] for g in groups])
        ax.set_title(f"Final {key} (last data point)")
        ax.grid(True, axis="y", alpha=0.3)
        for b, val in zip(bars, finals):
            ax.text(b.get_x() + b.get_width() / 2, val,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    fig_path = os.path.join(out_dir, "wave_final_bars.png")
    plt.savefig(fig_path, dpi=110)
    plt.close(fig)
    print(f"[+] saved {fig_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", type=str, default=None,
                    help="生成图表的目录 (默认: experiments/wave_seed_<seed>)")
    args = ap.parse_args()

    base = os.path.dirname(os.path.abspath(__file__))
    out_dir = args.out_dir or os.path.join(base, f"_analysis_seed_{args.seed}")
    os.makedirs(out_dir, exist_ok=True)

    print(f"[*] Analysis output: {out_dir}\n")

    metrics_by_group: Dict[str, Dict] = {}
    print("=" * 110)
    print("Per-group summary (last datapoint)")
    print("=" * 110)
    print(f"{'group':6s}  {'pts':5s}  {'rew':10s}  {'vfloss':10s}  {'pg':10s}  "
          f"{'sr':10s}  {'lam':10s}  {'vphys':10s}  {'vres':10s}  "
          f"{'td':10s}  {'tdres':10s}  {'adv':10s}")
    print("-" * 110)
    for group in ["A", "B", "C", "D"]:
        path = find_metrics(base, group, args.seed)
        if not path:
            print(f"{group:6s}  (no metrics.json found)")
            continue
        m = load_metrics(path)
        metrics_by_group[group] = m
        print(summary_row(group, m))

    print()
    print("=" * 110)
    print("Cross-group comparison (paper-targeted statements)")
    print("=" * 110)
    if {"A", "C"}.issubset(metrics_by_group):
        rA = metrics_by_group["A"].get("mean_reward", ([], []))[1]
        rC = metrics_by_group["C"].get("mean_reward", ([], []))[1]
        if rA and rC:
            print(f"  C vs A (PSC vs MLP under dense reward):  C={rC[-1]:.4f}  A={rA[-1]:.4f}  Δ={rC[-1] - rA[-1]:+.4f}")
    if {"B", "D"}.issubset(metrics_by_group):
        rB = metrics_by_group["B"].get("mean_reward", ([], []))[1]
        rD = metrics_by_group["D"].get("mean_reward", ([], []))[1]
        if rB and rD:
            print(f"  D vs B (PSC vs MLP under SPARSE reward): D={rD[-1]:.4f}  B={rB[-1]:.4f}  Δ={rD[-1] - rB[-1]:+.4f}")
    if {"C", "D"}.issubset(metrics_by_group):
        rC = metrics_by_group["C"].get("mean_reward", ([], []))[1]
        rD = metrics_by_group["D"].get("mean_reward", ([], []))[1]
        if rC and rD:
            print(f"  D vs C (information separation):         D={rD[-1]:.4f}  C={rC[-1]:.4f}  Δ={rD[-1] - rC[-1]:+.4f}")
    if {"A", "D"}.issubset(metrics_by_group):
        rA = metrics_by_group["A"].get("mean_reward", ([], []))[1]
        rD = metrics_by_group["D"].get("mean_reward", ([], []))[1]
        if rA and rD:
            print(f"  D vs A (full method vs traditional):     D={rD[-1]:.4f}  A={rA[-1]:.4f}  Δ={rD[-1] - rA[-1]:+.4f}")

    print()
    print("=" * 110)
    print("C2 evidence: TD-residual std reduction (PSC-only effect)")
    print("=" * 110)
    for group in ["A", "B", "C", "D"]:
        m = metrics_by_group.get(group, {})
        if "td_error_std" in m and "td_res_std" in m:
            tds = m["td_error_std"][1]
            tdrs = m["td_res_std"][1]
            if tds and tdrs:
                ratio = tdrs[-1] / tds[-1] if tds[-1] != 0 else float("nan")
                print(f"  {group}: td_error_std={tds[-1]:.4f}  td_res_std={tdrs[-1]:.4f}  ratio={ratio:.3f}")

    print()
    plot_group_curves(metrics_by_group, out_dir)


if __name__ == "__main__":
    main()
