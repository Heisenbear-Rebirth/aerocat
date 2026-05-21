"""
AeroCat v19.2 Ablation Runner — 4 group × N seed PSC ablation experiments

用法:
    # 单组单种子
    python -m aerocat.scripts.run_ablation --group D --seed 42

    # 短跑烟测 (全部 4 组 × 1 seed × 1M steps)
    python -m aerocat.scripts.run_ablation --group all --seeds 42 --total-timesteps 1000000

    # 完整实验 (4 组 × 5 seeds × 100M steps；约 20-40 小时)
    python -m aerocat.scripts.run_ablation --group all \
        --seeds 42 123 456 789 1024 --total-timesteps 100000000

实验组定义 (per NCA_FCS_v19.0_Design.md §6 Exp 1):
    A: use_psc=False, reward_type="dense"   ← 传统基线
    B: use_psc=False, reward_type="sparse"  ← 消融 (无 PSC + 稀疏 reward 应不收敛)
    C: use_psc=True,  reward_type="dense"   ← 消融 (信息双重编码)
    D: use_psc=True,  reward_type="sparse"  ← 完整方案

输出目录:
    v18/experiments/ablation_<group>_<reward>_<critic>/seed_<seed>/...
"""

import argparse
import os
import sys
from pathlib import Path

# 将 src/ 加入模块搜索路径
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# v19.2.2: 启用 JAX 持久化编译缓存 (节省重复运行时的 ~30s 编译)
_JAX_CACHE_DIR = os.environ.get(
    "JAX_COMPILATION_CACHE_DIR",
    str(Path(__file__).resolve().parents[3] / ".jax_cache"),
)
os.environ["JAX_COMPILATION_CACHE_DIR"] = _JAX_CACHE_DIR
os.makedirs(_JAX_CACHE_DIR, exist_ok=True)

from aerocat.config import TrainConfig, AblationConfig
from aerocat.training.ppo_trainer import make_train
import jax

# 显式启用 persistent compilation cache (推荐 JAX>=0.4.x)
try:
    from jax.experimental.compilation_cache import compilation_cache as cc
    cc.set_cache_dir(_JAX_CACHE_DIR)
except Exception:
    pass


GROUPS = ["A", "B", "C", "D", "E", "F"]   # E: fixed-w PSC; F: Cai 2025 dual-critic (v19.3)
DEFAULT_SEEDS = [42]


def build_config(group: str, seed: int, total_timesteps: int,
                 num_envs: int, num_steps: int,
                 base_log_dir: str,
                 critic_warmup_steps: int,
                 save_interval: int,
                 log_interval: int,
                 resume: bool = False,
                 task: str = "velocity",
                 psc_init_w=None,
                 psc_init_b=None,
                 init_tag: str = None,
                 disable_basis_idx: int = -1) -> TrainConfig:
    """为给定的消融组/种子构造完整 TrainConfig。

    所有非消融维度都保持完全一致以确保 4 组可比性。

    v19.2.1: 第一波诊断后调整若干默认值
      - critic_warmup_steps: 100 → 20 (PSC 已给 V_phys 锚点，长 warmup 浪费迭代)
      - save_interval: 200 → 10 (看到学习曲线，每 10 outer-iter 一个数据点)
      - log_interval: 10 → 5
      - total_timesteps default: 100M → 500M (Actor 真正学习需要更多 update)
    """
    config = TrainConfig()
    ablation = AblationConfig.from_group(group)
    # v19.3: 多任务支持 (默认 velocity = T1 兼容旧版)
    ablation.task = task
    # v19.4 init-sensitivity sweep wiring
    if psc_init_w is not None:
        ablation.psc_weights_init = tuple(psc_init_w)
    if psc_init_b is not None:
        ablation.psc_bias_init = float(psc_init_b)
    # v19.4 E2 per-basis leave-one-out
    if disable_basis_idx is not None and disable_basis_idx >= 0:
        if group.upper() != "D":
            print(f"[!] WARN: --disable-basis-idx is only meaningful for group D "
                  f"(PSC + sparse). Got group={group}. Proceeding anyway.")
        if disable_basis_idx not in (0, 1, 2, 3, 4):
            raise ValueError(f"--disable-basis-idx must be in 0..4, got {disable_basis_idx}")
        ablation.disable_basis_idx = int(disable_basis_idx)

    # 消融变量
    config.ablation = ablation

    # 公共训练参数
    config.seed = seed
    config.total_timesteps = total_timesteps
    config.num_envs = num_envs
    config.num_steps = num_steps

    # v19.2.1 ablation 调优（覆盖 TrainConfig 全局默认）
    config.ppo.critic_warmup_steps = critic_warmup_steps
    config.logging.save_interval = save_interval
    config.logging.log_interval = log_interval

    # 输出目录:
    #   experiments/ablation_<group>_<reward>_<critic>[_fixedw|_dual][_<task>]/seed_<seed>/
    # task=velocity 时不附后缀，保持 v19.2 目录与 1B 数据兼容
    critic_tag = "psc" if ablation.use_psc else "mlp"
    if ablation.fixed_psc_weights:
        critic_tag = critic_tag + "_fixedw"        # E 组目录区分
    if ablation.dual_critic:
        critic_tag = critic_tag + "_dual"          # F 组目录区分
    reward_tag = ablation.reward_type
    group_dir = f"ablation_{group}_{reward_tag}_{critic_tag}"
    if ablation.task != "velocity":
        group_dir = group_dir + f"_{ablation.task}"   # T2/T3 目录后缀
    if init_tag is not None:
        group_dir = group_dir + f"_init{init_tag}"    # v19.4 init sweep 目录后缀
    if ablation.disable_basis_idx >= 0:
        # v19.4 E2: per-basis leave-one-out directory suffix
        # ablation_D_sparse_psc_noPhi0/seed_<S>/  (Phi index in 0..4)
        group_dir = group_dir + f"_noPhi{ablation.disable_basis_idx}"
    seed_dir = f"seed_{seed}"

    full_log_dir = os.path.join(base_log_dir, group_dir, seed_dir)
    os.makedirs(full_log_dir, exist_ok=True)

    config.logging.log_dir = full_log_dir
    config.logging.experiment_name = f"{group_dir}_{seed_dir}"
    config.checkpoint.save_dir = os.path.join(full_log_dir, "checkpoints")

    # v19.2.1+: resume 支持。若指定 --resume，从已有 checkpoint 接着跑。
    if resume:
        ckpt_dir = os.path.join(full_log_dir, "checkpoints")
        if not os.path.isdir(ckpt_dir) or not os.listdir(ckpt_dir):
            raise FileNotFoundError(
                f"--resume specified but no checkpoint found at {ckpt_dir}"
            )
        config.checkpoint.resume_from = ckpt_dir
        # 找最大已保存步数以方便日志
        try:
            steps = [int(x) for x in os.listdir(ckpt_dir) if x.isdigit()]
            config._resume_from_step = max(steps) if steps else 0
        except Exception:
            config._resume_from_step = 0

    return config


def run_one(group: str, seed: int, args) -> None:
    """单组单种子训练。"""
    config = build_config(
        group=group,
        seed=seed,
        total_timesteps=args.total_timesteps,
        num_envs=args.num_envs,
        num_steps=args.num_steps,
        base_log_dir=args.experiments_dir,
        critic_warmup_steps=args.critic_warmup_steps,
        save_interval=args.save_interval,
        log_interval=args.log_interval,
        resume=args.resume,
        task=args.task,
        psc_init_w=args.psc_init_w,
        psc_init_b=args.psc_init_b,
        init_tag=args.init_tag,
        disable_basis_idx=args.disable_basis_idx,
    )

    print(f"\n{'=' * 70}")
    print(f"AeroCat v19.2 ABLATION — group={group} seed={seed}")
    print(f"{'=' * 70}")
    print(f"  use_psc:             {config.ablation.use_psc}")
    print(f"  reward_type:         {config.ablation.reward_type}")
    print(f"  fixed_psc_weights:   {config.ablation.fixed_psc_weights}")
    print(f"  dual_critic:         {config.ablation.dual_critic}")
    print(f"  task:                {config.ablation.task}")
    print(f"  group_label:         {config.ablation.group_label}")
    print(f"  total_timesteps:     {config.total_timesteps:,}")
    print(f"  num_envs:            {config.num_envs}")
    print(f"  num_steps:           {config.num_steps}")
    print(f"  batch_size:          {config.batch_size:,}")
    print(f"  num_updates:         {config.num_updates:,}")
    print(f"  critic_warmup_steps: {config.ppo.critic_warmup_steps}")
    print(f"  save_interval:       {config.logging.save_interval}")
    print(f"  log_dir:             {config.logging.log_dir}")
    if args.resume:
        from_step = getattr(config, "_resume_from_step", 0)
        remaining = config.total_timesteps - from_step
        print(f"  RESUME FROM:         {config.checkpoint.resume_from}")
        print(f"  resume_from_step:    {from_step:,}")
        print(f"  remaining_steps:     {remaining:,}")
    print(f"{'=' * 70}\n")

    # 持久化本次实验的 config 快照
    config_snapshot = os.path.join(config.logging.log_dir, "ablation_config.json")
    config.save(config_snapshot)
    print(f"[*] Config snapshot saved → {config_snapshot}")

    # 跑训练
    train_fn = make_train(config)
    rng = jax.random.PRNGKey(config.seed)
    final_state, metrics = train_fn(rng)

    print(f"\n[+] Group {group} seed {seed} finished.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AeroCat v19.2 PSC ablation runner")

    p.add_argument(
        "--group", type=str, required=True,
        help="实验组: A / B / C / D / all"
    )
    p.add_argument(
        "--seeds", type=int, nargs="+", default=DEFAULT_SEEDS,
        help="随机种子列表 (默认: 42)。完整实验建议: 42 123 456 789 1024"
    )
    p.add_argument(
        "--total-timesteps", type=int, default=500_000_000,
        help="每组训练总步数 (v19.2.1 默认: 500M)"
    )
    p.add_argument(
        "--num-envs", type=int, default=4096,
        help="并行环境数 (默认: 4096)"
    )
    p.add_argument(
        "--num-steps", type=int, default=128,
        help="每个环境的展开长度 (默认: 128)"
    )
    p.add_argument(
        "--experiments-dir", type=str,
        default=str(Path(__file__).resolve().parents[3] / "experiments"),
        help="实验输出根目录 (默认: v18/experiments/)"
    )
    p.add_argument(
        "--critic-warmup-steps", type=int, default=20,
        help="Critic warmup 迭代数 (v19.2.1 默认: 20，原 100 太长)"
    )
    p.add_argument(
        "--save-interval", type=int, default=10,
        help="metrics.json 保存间隔 (v19.2.1 默认: 10 outer-iter)"
    )
    p.add_argument(
        "--log-interval", type=int, default=5,
        help="终端日志打印间隔 (默认: 5)"
    )
    p.add_argument(
        "--device", type=str, default="gpu", choices=["cpu", "gpu"],
        help="JAX 设备"
    )
    p.add_argument(
        "--resume", action="store_true",
        help="从对应 group/seed 的 checkpoint 续训。需配合更大的 --total-timesteps "
             "(例如已跑 500M, 续训到 1B 用 --total-timesteps 1000000000 --resume)"
    )
    p.add_argument(
        "--task", type=str, default="velocity",
        choices=["velocity", "waypoint", "disturbance"],
        help="v19.3 多任务支持: velocity (T1 默认, v18-original), "
             "waypoint (T2, figure-8 P-controller), disturbance (T3, 时序扰动调度)"
    )
    p.add_argument(
        "--psc-init-w", type=float, nargs=5, default=None, metavar=("VEL", "ANG", "TILT", "INT", "SAT"),
        help="v19.4 init-sensitivity sweep: override the 5 PSC weight initial values "
             "(default = 45 2 2 0.5 1). Provide 5 floats. None preserves legacy default."
    )
    p.add_argument(
        "--psc-init-b", type=float, default=None,
        help="v19.4 init-sensitivity sweep: override PSC bias initial value "
             "(default = 20.0). None preserves legacy default."
    )
    p.add_argument(
        "--init-tag", type=str, default=None,
        help="v19.4: short tag appended to output dir to distinguish init configs "
             "(e.g. 'uniform' -> ablation_D_sparse_psc_initUniform/...). Required when "
             "--psc-init-w or --psc-init-b is set, to avoid overwriting baseline runs."
    )
    p.add_argument(
        "--disable-basis-idx", type=int, default=-1, choices=[-1, 0, 1, 2, 3, 4],
        help="v19.4 E2 per-basis leave-one-out: disable PSC basis at this index "
             "(0=vel_err, 1=omega, 2=tilt, 3=PID_integral, 4=saturation). "
             "Default -1 (no disable, bit-identical D group). Output dir gets _noPhiN suffix. "
             "Recommended only for --group D."
    )

    return p.parse_args()


def main():
    args = parse_args()

    if args.device == "cpu":
        jax.config.update("jax_platform_name", "cpu")

    if args.group.lower() == "all":
        groups = GROUPS
    else:
        groups = [args.group.upper()]
        if groups[0] not in GROUPS:
            print(f"[!] Unknown group: {args.group}; expected one of {GROUPS} or 'all'")
            sys.exit(1)

    print(f"[*] Ablation plan: groups={groups} × seeds={args.seeds}")
    print(f"[*] Total runs: {len(groups) * len(args.seeds)}")

    for group in groups:
        for seed in args.seeds:
            run_one(group, seed, args)

    print(f"\n[+] All ablation runs completed.")


if __name__ == "__main__":
    main()
