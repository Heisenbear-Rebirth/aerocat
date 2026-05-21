"""
DR / OOD Robustness Evaluation Script — Trajectory-Recording Variant (v19.4 follow-up)

Same evaluation protocol as `run_dr_eval.py` but additionally dumps per-step
trajectory data per (group, seed, lambda) cell, for C2 (conservatism mechanism)
and D1 (action smoothness) analyses.

Per-step record (batch-level, 256 parallel envs):
  - action[..., 0:4]     : (roll_rate, pitch_rate, yaw_rate, thrust), tanh-transformed, range [-1, 1]
  - saturation           : obs[..., 16], mixer-saturation diagnostic from L1
  - tilt                 : 2*(qx² + qy²), the crash criterion
  - v_err_norm           : ||v_cmd - v_actual||, heading-frame
  - active               : float mask, 1.0 if env is still in its current episode at this step

For each (group, seed, lambda) cell we save a single .npz file:
  traj_<group>_<seed>_<lambda>.npz
  containing per-step arrays of shape (n_steps, num_envs, *).

Aggregate stats (crash_rate / success_rate / etc.) are emitted into the same
results.json structure as run_dr_eval.py so existing analysis pipelines (P0)
still work.

Usage (from v18/):
    python -m aerocat.scripts.run_dr_eval_traj \\
        --groups A D \\
        --seeds 42 \\
        --lambdas 1.0 \\
        --num-episodes 20 \\
        --num-envs 256 \\
        --task velocity \\
        --output experiments/_c2d1/smoke.json \\
        --traj-dir experiments/_c2d1/smoke_trajectories

For a full re-run (after smoke goes well):
    --groups A B C D E F --seeds 42 123 456 789 1024 --lambdas 0.0 0.3 0.5 0.7 1.0 --num-episodes 100

Note: No training occurs. Reuses existing 1B-step checkpoints.
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

SAT_OBS_IDX = 16  # mixer saturation lives at obs[..., 16] per Section III-A description


def find_latest_checkpoint(group: str, seed: int, base_dir: str, task: str = "velocity") -> str:
    rt, ct = GROUP_DIR_MAP[group]
    task_suffix = "" if task == "velocity" else f"_{task}"
    ckpt_dir = os.path.join(base_dir, f"ablation_{group}_{rt}_{ct}{task_suffix}", f"seed_{seed}", "checkpoints")
    if not os.path.isdir(ckpt_dir):
        return ""
    return ckpt_dir


def make_env_config(curriculum_lambda: float, num_envs: int, ablation: "AblationConfig") -> tuple:
    cfg = TrainConfig()
    cfg.num_envs = num_envs
    cfg.num_steps = 128
    cfg.curriculum.lambda_start = curriculum_lambda
    cfg.ablation = ablation
    return cfg, cfg.get_runtime_env_config()


def evaluate_single(group: str, seed: int, curriculum_lambda: float,
                    num_episodes: int, num_envs: int, base_dir: str,
                    task: str = "velocity",
                    rng_seed: int = 12345,
                    traj_dir: str = None) -> dict:
    """Run num_episodes deterministic-policy evaluations at given lambda,
    additionally recording per-step trajectory tensors for C2/D1 analysis.

    If traj_dir is None, no trajectory file is written (identical behavior to run_dr_eval.py).
    """
    ablation = AblationConfig.from_group(group)
    ablation.task = task
    cfg, env_cfg = make_env_config(curriculum_lambda, num_envs, ablation)

    model, dummy_params, _ = create_stochastic_actor_critic(
        jax.random.PRNGKey(0), num_envs,
        use_psc=ablation.use_psc,
        fixed_psc_weights=ablation.fixed_psc_weights,
        dual_critic=ablation.dual_critic,
    )

    ckpt_dir = find_latest_checkpoint(group, seed, base_dir, task=task)
    if not ckpt_dir:
        return {"error": f"no checkpoint at {ckpt_dir}"}
    ckpt_mgr = CheckpointManager(ckpt_dir)
    restored, step = ckpt_mgr.restore_latest(item_structure=None)
    if restored is None or "params" not in restored:
        return {"error": f"failed to restore from {ckpt_dir}"}
    params = restored["params"]

    rng = jax.random.PRNGKey(rng_seed)
    rng, init_rng, env_rng = jax.random.split(rng, 3)
    env_params = generate_params(init_rng, num_envs, cfg.physics, curriculum_lambda)
    state, _, obs = reset_env(env_rng, num_envs, env_cfg, curriculum_lambda, params=env_params)
    lstm_state = model.init_lstm_state(num_envs)

    @jax.jit
    def det_step(state, env_params, lstm_state, obs, rng_step):
        action_mean, action_std, _, new_lstm = model.apply(params, obs, lstm_state, None)
        action = jnp.tanh(action_mean)
        new_state, timestep, new_env_params = step_env(
            rng_step, state, action, env_params, env_cfg, curriculum_lambda
        )
        done_mask = timestep.done[:, None].astype(jnp.float32)
        new_lstm = type(new_lstm)(
            h=new_lstm.h * (1.0 - done_mask),
            c=new_lstm.c * (1.0 - done_mask),
        )
        return new_state, new_env_params, new_lstm, timestep, action

    # Stats accumulators (same as run_dr_eval.py)
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

    # Trajectory buffers (per-step, batch-level)
    record_traj = traj_dir is not None
    traj_action = []      # list of (num_envs, 4)
    traj_sat = []         # list of (num_envs,)
    traj_tilt = []        # list of (num_envs,)
    traj_v_err = []       # list of (num_envs,)
    traj_active = []      # list of (num_envs,) — 1.0 if env is mid-episode pre-step

    max_steps_per_episode = int(cfg.env.max_episode_time / 0.02)
    target_completed = num_episodes
    step_count = 0
    max_iter = max_steps_per_episode * 4

    # Track per-env episode index assignment for "active" mask
    active_mask = jnp.ones(num_envs, dtype=jnp.float32)

    while completed_episodes < target_completed and step_count < max_iter:
        rng, step_rng = jax.random.split(rng)

        # Record pre-step observables for the trajectory (active envs)
        if record_traj:
            sat_pre = np.asarray(obs[..., SAT_OBS_IDX])
            qx_pre = np.asarray(state.phys_state.quaternion[..., 1])
            qy_pre = np.asarray(state.phys_state.quaternion[..., 2])
            tilt_pre = 2.0 * (qx_pre ** 2 + qy_pre ** 2)
            v_err_pre = np.linalg.norm(np.asarray(state.l1_state.velocity_error_heading), axis=-1)
            active_pre = np.asarray(active_mask)

        state, env_params, lstm_state, timestep, action = det_step(
            state, env_params, lstm_state, obs, step_rng
        )

        if record_traj:
            traj_action.append(np.asarray(action))
            traj_sat.append(sat_pre)
            traj_tilt.append(tilt_pre)
            traj_v_err.append(v_err_pre)
            traj_active.append(active_pre)

        obs = timestep.obs
        success_steps += int(jnp.sum(timestep.reward > 0.5))
        total_steps += num_envs
        episode_returns = episode_returns + timestep.reward
        episode_lengths = episode_lengths + 1.0

        v_err = state.l1_state.velocity_error_heading
        v_err_norm_sq = jnp.sum(v_err ** 2, axis=-1)
        v_err_sq_sum += float(jnp.sum(v_err_norm_sq))
        v_err_sq_n += num_envs

        qx = state.phys_state.quaternion[..., 1]
        qy = state.phys_state.quaternion[..., 2]
        tilt = 2.0 * (qx ** 2 + qy ** 2)
        crashed_now = tilt > 1.5
        crashed_count += int(jnp.sum(crashed_now & timestep.done))

        done_now = jnp.array(timestep.done)
        if jnp.any(done_now):
            completed_returns.extend([float(r) for r, d in zip(episode_returns, done_now) if d])
            completed_lengths.extend([float(l) for l, d in zip(episode_lengths, done_now) if d])
            completed_episodes += int(jnp.sum(done_now))
            episode_returns = episode_returns * (1.0 - done_now.astype(jnp.float32))
            episode_lengths = episode_lengths * (1.0 - done_now.astype(jnp.float32))

            # An env that just `done` will start a new episode at the *next* step; mark inactive
            # for current-step (its post-step state was for the boundary, not a new episode).
            # Active mask is for the *next* iteration's pre-step.
            active_mask = jnp.where(done_now, 1.0, active_mask)
            # All envs always immediately auto-reset in this loop, so they stay active=1.0.

        step_count += 1

    completed_returns = completed_returns[:target_completed]
    completed_lengths = completed_lengths[:target_completed]

    # Write trajectory file if recording
    if record_traj and len(traj_action) > 0:
        os.makedirs(traj_dir, exist_ok=True)
        traj_path = os.path.join(traj_dir, f"traj_{group}_{seed}_lam{curriculum_lambda:.1f}.npz")
        np.savez_compressed(
            traj_path,
            action=np.asarray(traj_action),        # (n_steps, num_envs, 4)
            saturation=np.asarray(traj_sat),       # (n_steps, num_envs)
            tilt=np.asarray(traj_tilt),            # (n_steps, num_envs)
            v_err=np.asarray(traj_v_err),          # (n_steps, num_envs)
            active=np.asarray(traj_active),        # (n_steps, num_envs)
            meta={  # only readable via allow_pickle=True; but stored as object array
                "group": group, "seed": seed, "lambda": float(curriculum_lambda),
                "num_envs": num_envs, "n_steps": len(traj_action),
            },
        )

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
        "checkpoint_step":     int(step) if step is not None else 0,
        "traj_file":           traj_path if record_traj and len(traj_action) > 0 else None,
        "n_traj_steps":        len(traj_action) if record_traj else 0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", nargs="+", default=["A", "B", "C", "D", "E", "F"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456, 789, 1024])
    ap.add_argument("--lambdas", type=float, nargs="+", default=[0.0, 0.3, 0.5, 0.7, 1.0])
    ap.add_argument("--num-episodes", type=int, default=100)
    ap.add_argument("--num-envs", type=int, default=256)
    ap.add_argument("--task", type=str, default="velocity",
                    choices=["velocity", "waypoint", "disturbance"])
    ap.add_argument("--base-dir", type=str,
                    default=str(Path(__file__).resolve().parents[3] / "experiments"))
    ap.add_argument("--output", type=str,
                    default=str(Path(__file__).resolve().parents[3] / "experiments" / "_c2d1" / "results.json"))
    ap.add_argument("--traj-dir", type=str,
                    default=str(Path(__file__).resolve().parents[3] / "experiments" / "_c2d1" / "trajectories"),
                    help="Directory for per-cell trajectory .npz files. Use empty string to disable.")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    traj_dir = args.traj_dir if args.traj_dir else None
    if traj_dir:
        os.makedirs(traj_dir, exist_ok=True)

    print(f"[*] trajectory-recording OOD eval — task={args.task}")
    print(f"[*] groups={args.groups} seeds={args.seeds} lambdas={args.lambdas}")
    print(f"[*] {args.num_episodes} ep/cell × {args.num_envs} envs;  "
          f"total cells = {len(args.groups) * len(args.seeds) * len(args.lambdas)}")
    print(f"[*] traj_dir = {traj_dir}")
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
                        task=args.task, traj_dir=traj_dir,
                    )
                except Exception as e:
                    r = {"group": g, "seed": s, "lambda": lam, "error": str(e)}
                done += 1
                dt = time.time() - t1
                eta_sec = (time.time() - t_start) / done * (total - done)
                err_str = f"  ERROR: {r.get('error', '')}" if "error" in r else ""
                print(f"[{done}/{total}] G={g} seed={s:>5d} λ={lam:.1f}  "
                      f"crash={r.get('crash_rate', 0):.2%}  traj_steps={r.get('n_traj_steps', 0)}  "
                      f"({dt:.1f}s, ETA {eta_sec/60:.1f} min){err_str}")
                all_results.append(r)
                with open(args.output, "w") as f:
                    json.dump(all_results, f, indent=2)

    print(f"\n[+] Done in {(time.time()-t_start)/60:.1f} min. Aggregate -> {args.output}")
    if traj_dir:
        print(f"[+] Trajectory files -> {traj_dir}")


if __name__ == "__main__":
    main()
