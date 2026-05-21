"""
DR / OOD Robustness Evaluation Script (v19.3)

Loads pre-trained 1B-step checkpoints for groups A/B/C/D/E (and F when ready)
and evaluates them at multiple curriculum_lambda levels (DR strengths) without
any further training.

Reports per-(group, seed, lambda) statistics:
  - mean_episode_reward
  - tracking_rmse (velocity error norm)
  - episode_length (avg steps before timeout/crash)
  - crash_rate
  - success_rate (fraction of steps with reward > 0.5)

Usage:
    python -m aerocat.scripts.run_dr_eval \\
        --groups A B C D E \\
        --seeds 42 123 456 789 1024 \\
        --lambdas 0.0 0.3 0.5 0.7 1.0 \\
        --num-episodes 100 \\
        --num-envs 256 \\
        --output experiments/_dr_eval/results.json

Note: No training occurs. Reuses existing checkpoints from
      experiments/ablation_<G>_<reward>_<critic>[_fixedw|_dual]/seed_<S>/checkpoints/
"""
import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

# Path setup
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# JAX cache (skip recompile across groups)
_JAX_CACHE_DIR = os.environ.get(
    "JAX_COMPILATION_CACHE_DIR",
    str(Path(__file__).resolve().parents[3] / ".jax_cache"),
)
os.environ["JAX_COMPILATION_CACHE_DIR"] = _JAX_CACHE_DIR
os.makedirs(_JAX_CACHE_DIR, exist_ok=True)

import jax
import jax.numpy as jnp
import numpy as np

from aerocat.config import TrainConfig, AblationConfig
from aerocat.networks.actor_critic import (
    create_stochastic_actor_critic, OBS_DIM, ACTION_DIM
)
from aerocat.envs.uav_env import reset_env, step_env, EnvConfig
from aerocat.generators.param_generator import init_params as generate_params
from aerocat.utils.checkpoint_manager import CheckpointManager


GROUP_DIR_MAP = {
    "A": ("dense", "mlp"),
    "B": ("sparse", "mlp"),
    "C": ("dense", "psc"),
    "D": ("sparse", "psc"),
    "E": ("sparse", "psc_fixedw"),
    "F": ("dense", "mlp_dual"),
}


def find_latest_checkpoint(group: str, seed: int, base_dir: str, task: str = "velocity") -> str:
    rt, ct = GROUP_DIR_MAP[group]
    # v19.3 multi-task dir naming: task=velocity has no suffix; others get _<task>
    task_suffix = "" if task == "velocity" else f"_{task}"
    ckpt_dir = os.path.join(base_dir, f"ablation_{group}_{rt}_{ct}{task_suffix}", f"seed_{seed}", "checkpoints")
    if not os.path.isdir(ckpt_dir):
        return ""
    return ckpt_dir


def make_env_config(curriculum_lambda: float, num_envs: int, ablation: "AblationConfig") -> tuple:
    """Build runtime env config + train config snapshot for given lambda.

    The ablation (including .task and .reward_type) must be applied BEFORE
    get_runtime_env_config() because env-side reward/task wiring depends on it.
    """
    cfg = TrainConfig()
    cfg.num_envs = num_envs
    cfg.num_steps = 128                          # not used here, but kept for compat
    cfg.curriculum.lambda_start = curriculum_lambda
    cfg.ablation = ablation                       # MUST be set before get_runtime_env_config()
    return cfg, cfg.get_runtime_env_config()


def evaluate_single(group: str, seed: int, curriculum_lambda: float,
                    num_episodes: int, num_envs: int, base_dir: str,
                    task: str = "velocity",
                    rng_seed: int = 12345) -> dict:
    """Run num_episodes deterministic-policy evaluations at given lambda.

    Returns aggregate stats dict.
    """
    ablation = AblationConfig.from_group(group)
    ablation.task = task
    cfg, env_cfg = make_env_config(curriculum_lambda, num_envs, ablation)

    # Build network with matching ablation config
    model, dummy_params, _ = create_stochastic_actor_critic(
        jax.random.PRNGKey(0), num_envs,
        use_psc=ablation.use_psc,
        fixed_psc_weights=ablation.fixed_psc_weights,
        dual_critic=ablation.dual_critic,
    )

    # Load checkpoint. Saved checkpoints contain the full training state
    # (params, opt_state, env_state, lstm_state, rng, step, ...), so we
    # restore as raw pytree (structure=None) and pull out 'params'.
    ckpt_dir = find_latest_checkpoint(group, seed, base_dir, task=task)
    if not ckpt_dir:
        return {"error": f"no checkpoint at {ckpt_dir}"}
    ckpt_mgr = CheckpointManager(ckpt_dir)
    restored, step = ckpt_mgr.restore_latest(item_structure=None)
    if restored is None or "params" not in restored:
        return {"error": f"failed to restore from {ckpt_dir}"}
    params = restored["params"]

    # Reset environment with given lambda
    rng = jax.random.PRNGKey(rng_seed)
    rng, init_rng, env_rng = jax.random.split(rng, 3)
    env_params = generate_params(init_rng, num_envs, cfg.physics, curriculum_lambda)
    state, _, obs = reset_env(env_rng, num_envs, env_cfg, curriculum_lambda, params=env_params)
    lstm_state = model.init_lstm_state(num_envs)

    # JIT-compiled deterministic step (action_mean, no noise)
    @jax.jit
    def det_step(state, env_params, lstm_state, obs, rng_step):
        action_mean, action_std, _, new_lstm = model.apply(params, obs, lstm_state, None)
        action = jnp.tanh(action_mean)   # deterministic; matches sample_action's tanh transform
        new_state, timestep, new_env_params = step_env(
            rng_step, state, action, env_params, env_cfg, curriculum_lambda
        )
        # Reset LSTM on done (consistent with rollout)
        done_mask = timestep.done[:, None].astype(jnp.float32)
        new_lstm = type(new_lstm)(
            h=new_lstm.h * (1.0 - done_mask),
            c=new_lstm.c * (1.0 - done_mask),
        )
        return new_state, new_env_params, new_lstm, timestep

    # Stats accumulators
    rewards_per_step = []
    crashed_count = 0
    success_steps = 0
    total_steps = 0
    v_err_sq_sum = 0.0
    v_err_sq_n = 0
    episode_returns = jnp.zeros(num_envs)
    episode_lengths = jnp.zeros(num_envs)
    completed_episodes = 0
    completed_returns = []
    completed_lengths = []

    max_steps_per_episode = int(cfg.env.max_episode_time / 0.02)  # 50Hz
    target_completed = num_episodes
    step_count = 0
    max_iter = max_steps_per_episode * 4   # 4x safety margin

    while completed_episodes < target_completed and step_count < max_iter:
        rng, step_rng = jax.random.split(rng)
        state, env_params, lstm_state, timestep = det_step(
            state, env_params, lstm_state, obs, step_rng
        )
        obs = timestep.obs
        rewards_per_step.append(float(jnp.mean(timestep.reward)))
        success_steps += int(jnp.sum(timestep.reward > 0.5))
        total_steps += num_envs
        episode_returns = episode_returns + timestep.reward
        episode_lengths = episode_lengths + 1.0

        # Track velocity error (heading-frame speed error norm)
        v_err = state.l1_state.velocity_error_heading  # [batch, 3]
        v_err_norm_sq = jnp.sum(v_err ** 2, axis=-1)
        v_err_sq_sum += float(jnp.sum(v_err_norm_sq))
        v_err_sq_n += num_envs

        # Crash detection (tilt > 1.5 ≈ 90°)
        qx = state.phys_state.quaternion[..., 1]
        qy = state.phys_state.quaternion[..., 2]
        tilt = 2.0 * (qx ** 2 + qy ** 2)
        crashed_now = tilt > 1.5
        crashed_count += int(jnp.sum(crashed_now & timestep.done))

        # Collect completed episodes
        done_now = jnp.array(timestep.done)
        if jnp.any(done_now):
            completed_returns.extend([float(r) for r, d in zip(episode_returns, done_now) if d])
            completed_lengths.extend([float(l) for l, d in zip(episode_lengths, done_now) if d])
            completed_episodes += int(jnp.sum(done_now))
            # Reset trackers for done envs
            episode_returns = episode_returns * (1.0 - done_now.astype(jnp.float32))
            episode_lengths = episode_lengths * (1.0 - done_now.astype(jnp.float32))

        step_count += 1

    # Trim to exactly target_completed (extras may exist due to batch wrap)
    completed_returns = completed_returns[:target_completed]
    completed_lengths = completed_lengths[:target_completed]

    return {
        "group": group,
        "seed": seed,
        "lambda": curriculum_lambda,
        "n_episodes": len(completed_returns),
        "mean_episode_return": float(np.mean(completed_returns)) if completed_returns else 0.0,
        "sd_episode_return":   float(np.std(completed_returns))  if completed_returns else 0.0,
        "mean_episode_length": float(np.mean(completed_lengths)) if completed_lengths else 0.0,
        "crash_rate":          crashed_count / max(target_completed, 1),
        "success_rate":        success_steps / max(total_steps, 1),
        "tracking_rmse":       float(np.sqrt(v_err_sq_sum / max(v_err_sq_n, 1))),
        "checkpoint_step":     int(step) if isinstance(step, (int, np.integer)) else int(step) if step is not None else 0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", nargs="+", default=["A", "B", "C", "D", "E"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456, 789, 1024])
    ap.add_argument("--lambdas", type=float, nargs="+", default=[0.0, 0.3, 0.5, 0.7, 1.0])
    ap.add_argument("--num-episodes", type=int, default=100)
    ap.add_argument("--num-envs", type=int, default=256)
    ap.add_argument("--task", type=str, default="velocity",
                    choices=["velocity", "waypoint", "disturbance"],
                    help="v19.3 multi-task: velocity=T1, waypoint=T2, disturbance=T3")
    ap.add_argument("--base-dir", type=str,
                    default=str(Path(__file__).resolve().parents[3] / "experiments"))
    ap.add_argument("--output", type=str,
                    default=str(Path(__file__).resolve().parents[3] / "experiments" / "_dr_eval" / "results.json"))
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    print(f"[*] DR/OOD eval — task={args.task} groups={args.groups}, seeds={args.seeds}, lambdas={args.lambdas}")
    print(f"[*] {args.num_episodes} episodes per (group, seed, lambda) "
          f"with {args.num_envs} parallel envs")
    print(f"[*] Total evaluations: {len(args.groups) * len(args.seeds) * len(args.lambdas)}")
    print()

    all_results = []
    t_start = time.time()
    total = len(args.groups) * len(args.seeds) * len(args.lambdas)
    done = 0

    for g in args.groups:
        for s in args.seeds:
            for lam in args.lambdas:
                t1 = time.time()
                try:
                    r = evaluate_single(
                        g, s, lam, args.num_episodes, args.num_envs, args.base_dir,
                        task=args.task,
                    )
                except Exception as e:
                    r = {"group": g, "seed": s, "lambda": lam, "error": str(e)}
                done += 1
                dt = time.time() - t1
                eta_sec = (time.time() - t_start) / done * (total - done)
                err_str = f"  ERROR: {r.get('error', '')}" if "error" in r else ""
                print(f"[{done}/{total}] G={g} seed={s:>5d} λ={lam:.1f}  "
                      f"return={r.get('mean_episode_return', 0):.3f}  "
                      f"crash={r.get('crash_rate', 0):.2%}  "
                      f"({dt:.1f}s, ETA {eta_sec/60:.1f} min){err_str}")
                all_results.append(r)
                # Save incrementally
                with open(args.output, "w") as f:
                    json.dump(all_results, f, indent=2)

    print(f"\n[+] Done in {(time.time()-t_start)/60:.1f} min. Results -> {args.output}")


if __name__ == "__main__":
    main()
