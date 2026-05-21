"""
AeroCat v19.0 Training Entry Point

用法:
  # 从默认 config 启动新训练
  python train.py

  # 从 checkpoint 恢复训练
  python train.py --resume logs/aerocat_ppo_20260411_115205/checkpoints

  # 使用自定义 config
  python train.py --config my_config.json

  # 恢复 + 自定义 config
  python train.py --config my_config.json --resume logs/aerocat_ppo_20260411_115205/checkpoints
"""

import argparse
import sys
import os
from pathlib import Path

# 将 src/ 加入模块搜索路径，使 `aerocat` 包可被导入
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from aerocat.config import TrainConfig
from aerocat.training.ppo_trainer import make_train
import jax


def find_latest_log_checkpoints(log_dir: str = "logs") -> str:
    """自动找到 logs/ 下最新一次训练的 checkpoints 目录"""
    if not os.path.exists(log_dir):
        return None
    subdirs = sorted(
        [d for d in os.listdir(log_dir) if d.startswith("aerocat_ppo_")],
        reverse=True
    )
    for d in subdirs:
        ckpt_path = os.path.join(log_dir, d, "checkpoints")
        if os.path.exists(ckpt_path) and os.listdir(ckpt_path):
            return ckpt_path
    return None


def main():
    parser = argparse.ArgumentParser(description="AeroCat PPO Training")
    parser.add_argument("--config", type=str, default=None,
                        help="JSON 配置文件路径 (默认使用内置 TrainConfig)")
    parser.add_argument("--resume", type=str, nargs="?", const="auto",
                        help="从 checkpoint 恢复。不指定路径则自动查找最新 checkpoint")
    parser.add_argument("--device", type=str, default="gpu", choices=["cpu", "gpu"],
                        help="JAX 设备")

    args = parser.parse_args()

    # 配置 JAX 设备
    if args.device == "cpu":
        jax.config.update("jax_platform_name", "cpu")

    # 1. 加载配置
    if args.config:
        config_path = os.path.abspath(args.config)
        if not os.path.exists(config_path):
            print(f"[!] 配置文件不存在: {config_path}")
            sys.exit(1)
        config = TrainConfig.load(config_path)
        print(f"[*] 从配置文件加载: {config_path}")
    else:
        config = TrainConfig()
        print("[*] 使用默认配置")

    # 2. 处理恢复
    if args.resume is not None:
        if args.resume == "auto":
            resume_path = find_latest_log_checkpoints(config.logging.log_dir)
            if resume_path:
                print(f"[*] 自动检测到最新 checkpoint: {resume_path}")
            else:
                print("[!] 未找到可恢复的 checkpoint，将从头开始训练")
        else:
            resume_path = os.path.abspath(args.resume)
        
        if resume_path and os.path.exists(resume_path):
            config.checkpoint.resume_from = resume_path
            print(f"[*] 将从 {resume_path} 恢复训练")
        elif args.resume != "auto":
            print(f"[!] 恢复路径不存在: {resume_path}")
            sys.exit(1)

    # 3. 打印关键配置
    print(f"\n{'=' * 50}")
    print(f"AeroCat Training Configuration")
    print(f"{'=' * 50}")
    print(f"  seed:            {config.seed}")
    print(f"  num_envs:        {config.num_envs}")
    print(f"  total_timesteps: {config.total_timesteps:,}")
    print(f"  num_steps:       {config.num_steps}")
    print(f"  batch_size:      {config.batch_size:,}")
    print(f"  num_updates:     {config.num_updates:,}")
    print(f"  lr:              {config.ppo.learning_rate}")
    print(f"  clip_eps:        {config.ppo.clip_eps}")
    print(f"  vf_coef:         {config.ppo.vf_coef}")
    print(f"  warmup_steps:    {config.ppo.critic_warmup_steps}")
    print(f"  resume_from:     {config.checkpoint.resume_from or 'None (fresh start)'}")
    print(f"{'=' * 50}\n")

    # 4. 保存本次使用的 config
    config.save(os.path.join(config.logging.log_dir, "last_config.json"))

    # 5. 创建并运行训练
    train_fn = make_train(config)
    rng = jax.random.PRNGKey(config.seed)
    final_state, metrics = train_fn(rng)

    print("Training finished!")


if __name__ == "__main__":
    main()
