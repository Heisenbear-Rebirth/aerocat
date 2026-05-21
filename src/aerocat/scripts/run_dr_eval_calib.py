"""
DR / OOD Robustness Evaluation — V calibration variant (v19.4 C1)

Identical eval protocol as run_dr_eval.py (deterministic policy), but additionally
dumps per-step (value, v_phys, v_res, reward, done, active) into per-cell .npz
files for offline calibration analysis.

The analysis question is: does PSC's analytic V_phys baseline keep the critic
better-calibrated in OOD states (where V_res alone is extrapolating)?

For each (group, seed, lambda) cell we save:
    traj_<group>_<seed>_lam<L>.npz
    arrays of shape (n_steps, num_envs):
      value, v_phys, v_res, reward, done, active

Aggregate stats (crash_rate / success_rate / ...) are also emitted into a JSON
sharing schema with run_dr_eval.py for cross-validation.

Usage (from v18/):
    python src/aerocat/scripts/run_dr_eval_calib.py \\
        --groups A B C D E F --seeds 42 123 456 789 1024 \\
        --lambdas 0.0 0.3 0.5 0.7 1.0 --num-episodes 100 --num-envs 256 \\
        --task velocity --output experiments/_c1/T1_calib_results.json \\
        --traj-dir experiments/_c1/T1_calib_trajectories
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

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
from aerocat.networks.actor_critic import create_stochastic_actor_critic, OBS_DIM, ACTION_DIM
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
    task_suffix = "" if task == "velocity" else f"_{task}"
    ckpt_dir = os.path.join(base_dir, f"ablation_{group}_{rt}_{ct}{task_suffix}",
                            f"seed_{seed}", "checkpoints")
    if not os.path.isdir(ckpt_dir):
        return ""
    return ckpt_dir


def make_env_config(curriculum_lambda: float, num_envs: int, ablation: "AblationConfig"):
    cfg = TrainConfig()
    cfg.num_envs = num_envs
    cfg.num_steps = 128
    cfg.curriculum.lambda_start = curriculum_lambda
    cfg.ablation = ablation
    return cfg, cfg.get_runtime_env_config()


def evaluate_single(group, seed, curriculum_lambda, num_episodes, num_envs,
                    base_dir, task="velocity", rng_seed=12345, traj_dir=None):
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

    # Deterministic step + V/V_phys/V_res dump
    @jax.jit
    def det_step(state, env_params, lstm_state, obs, rng_step):
        action_mean, action_std, value_main, new_lstm = model.apply(
            params, obs, lstm_state, None
        )
        # Also call critic_forward to decompose into v_phys / v_res
        v_total, v_phys, v_res = model.apply(
            params, obs, method=model.critic_forward
        )
        action = jnp.tanh(action_mean)
        new_state, timestep, new_env_params = step_env(
            rng_step, state, action, env_params, env_cfg, curriculum_lambda
        )
        done_mask = timestep.done[:, None].astype(jnp.float32)
        new_lstm = type(new_lstm)(
            h=new_lstm.h * (1.0 - done_mask),
            c=new_lstm.c * (1.0 - done_mask),
        )
        return new_state, new_env_params, new_lstm, timestep, v_total, v_phys, v_res

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

    record = traj_dir is not None
    tr_value = []
    tr_v_phys = []
    tr_v_res = []
    tr_reward = []
    tr_done = []
    tr_active = []

    max_steps_per_episode = int(cfg.env.max_episode_time / 0.02)
    target_completed = num_episodes
    step_count = 0
    max_iter = max_steps_per_episode * 4

    active_mask = jnp.ones(num_envs, dtype=jnp.float32)

    while completed_episodes < target_completed and step_count < max_iter:
        rng, step_rng = jax.random.split(rng)

        if record:
            active_pre = np.asarray(active_mask)

        state, env_params, lstm_state, timestep, v_total, v_phys, v_res = det_step(
            state, env_params, lstm_state, obs, step_rng
        )

        if record:
            tr_value.append(np.asarray(v_total))
            tr_v_phys.append(np.asarray(v_phys))
            tr_v_res.append(np.asarray(v_res))
            tr_reward.append(np.asarray(timestep.reward))
            tr_done.append(np.asarray(timestep.done))
            tr_active.append(active_pre)

        obs = timestep.obs
        success_steps += int(jnp.sum(timestep.reward > 0.5))
        total_steps += num_envs
        episode_returns = episode_returns + timestep.reward
        episode_lengths = episode_lengths + 1.0

        v_err = state.l1_state.velocity_error_heading
        v_err_sq_sum += float(jnp.sum(jnp.sum(v_err ** 2, axis=-1)))
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

        step_count += 1

    completed_returns = completed_returns[:target_completed]
    completed_lengths = completed_lengths[:target_completed]

    traj_path = None
    if record and len(tr_value) > 0:
        os.makedirs(traj_dir, exist_ok=True)
        traj_path = os.path.join(traj_dir, f"traj_{group}_{seed}_lam{curriculum_lambda:.1f}.npz")
        np.savez_compressed(
            traj_path,
            value=np.asarray(tr_value),
            v_phys=np.asarray(tr_v_phys),
            v_res=np.asarray(tr_v_res),
            reward=np.asarray(tr_reward),
            done=np.asarray(tr_done),
            active=np.asarray(tr_active),
        )

    return {
        "group": group,
        "seed": seed,
        "lambda": curriculum_lambda,
        "policy": "deterministic",
        "n_episodes": len(completed_returns),
        "mean_episode_return": float(np.mean(completed_returns)) if completed_returns else 0.0,
        "sd_episode_return":   float(np.std(completed_returns))  if completed_returns else 0.0,
        "mean_episode_length": float(np.mean(completed_lengths)) if completed_lengths else 0.0,
        "crash_rate":          crashed_count / max(target_completed, 1),
        "success_rate":        success_steps / max(total_steps, 1),
        "tracking_rmse":       float(np.sqrt(v_err_sq_sum / max(v_err_sq_n, 1))),
        "checkpoint_step":     int(step) if step is not None else 0,
        "traj_file":           traj_path,
        "n_traj_steps":        len(tr_value),
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
                    default=str(Path(__file__).resolve().parents[3] / "experiments" / "_c1" / "results.json"))
    ap.add_argument("--traj-dir", type=str,
                    default=str(Path(__file__).resolve().parents[3] / "experiments" / "_c1" / "trajectories"))
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    traj_dir = args.traj_dir if args.traj_dir else None
    if traj_dir:
        os.makedirs(traj_dir, exist_ok=True)

    print(f"[*] V-calibration OOD eval (deterministic) — task={args.task}")
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
                    r = evaluate_single(g, s, lam, args.num_episodes, args.num_envs,
                                        args.base_dir, task=args.task, traj_dir=traj_dir)
                except Exception as e:
                    r = {"group": g, "seed": s, "lambda": lam, "error": str(e)}
                done += 1
                dt = time.time() - t1
                eta_sec = (time.time() - t_start) / done * (total - done)
                err_str = f"  ERROR: {r.get('error', '')}" if "error" in r else ""
                print(f"[{done}/{total}] G={g} seed={s:>5d} λ={lam:.1f}  "
                      f"crash={r.get('crash_rate', 0):.2%}  steps={r.get('n_traj_steps', 0)}  "
                      f"({dt:.1f}s, ETA {eta_sec/60:.1f} min){err_str}")
                all_results.append(r)
                with open(args.output, "w") as f:
                    json.dump(all_results, f, indent=2)

    print(f"\n[+] Done in {(time.time()-t_start)/60:.1f} min. Aggregate -> {args.output}")
    if traj_dir:
        print(f"[+] Trajectory files -> {traj_dir}")


if __name__ == "__main__":
    main()
