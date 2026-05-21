"""
H5 — Per-dimension OOD perturbation eval (v19.4).

Decomposes the monolithic λ=1.0 OOD perturbation into 7 named dimensions and runs
deterministic eval with ONE dim at λ=1 and all others at λ=0 ("nominal"). Plus a
nominal baseline and an all-λ=1 sanity check that should match P0 λ=1 results.

Strategy:
  1. Sample baseline params at curriculum_lambda=0.0  →  phys_easy
  2. Sample hard params     at curriculum_lambda=1.0  →  phys_hard
  3. For each dim D, override SPECIFIC phys_params fields in phys_easy with the
     corresponding fields from phys_hard.
  4. Reset env with init_lambda (controls max_tilt, max_rate, replay window — i.e.
     initial-state difficulty).
  5. Step env with step_lambda (controls collision-event probability inside
     step_env via collision_event() in uav_env.py).

Dims:
  nominal      — phys_easy, init λ=0, step λ=0           (baseline)
  mass         — override mass/inertia/cog_offset       (geometry+inertia hard)
  wind         — override wind_speed_mean/wind_direction
  turb         — override turbulence_intensity + all gust_* fields
  sensor       — override gyro_bias + accel_bias (only biases scale with λ)
  actuator     — override motor_loss
  init_state   — phys_easy, init λ=1, step λ=0          (hard initial tilt/rate)
  collision    — phys_easy, init λ=0, step λ=1          (random collision shocks)
  all_l1       — phys_hard,  init λ=1, step λ=1          (sanity = full λ=1)

Usage:
  python src/aerocat/scripts/run_dr_eval_perdim.py \\
      --groups A B C D E F --seeds 42 123 456 789 1024 \\
      --dims nominal mass wind turb sensor actuator init_state collision all_l1 \\
      --num-episodes 100 --num-envs 256 \\
      --output experiments/_h5_perdim/results.json
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
from aerocat.networks.actor_critic import create_stochastic_actor_critic
from aerocat.envs.uav_env import reset_env, step_env
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

DIM_FIELDS = {
    "mass":      ["mass", "inertia_xx", "inertia_yy", "inertia_zz", "cog_offset"],
    "wind":      ["wind_speed_mean", "wind_direction"],
    "turb":      ["turbulence_intensity", "gust_active", "gust_start_time",
                  "gust_duration", "gust_magnitude", "gust_direction"],
    "sensor":    ["gyro_bias", "accel_bias"],
    "actuator":  ["motor_loss"],
}

# (init_lambda, step_lambda, use_phys_hard, fields_from_hard)
DIM_CONFIG = {
    "nominal":    (0.0, 0.0, False, []),
    "mass":       (0.0, 0.0, False, DIM_FIELDS["mass"]),
    "wind":       (0.0, 0.0, False, DIM_FIELDS["wind"]),
    "turb":       (0.0, 0.0, False, DIM_FIELDS["turb"]),
    "sensor":     (0.0, 0.0, False, DIM_FIELDS["sensor"]),
    "actuator":   (0.0, 0.0, False, DIM_FIELDS["actuator"]),
    "init_state": (1.0, 0.0, False, []),
    "collision":  (0.0, 1.0, False, []),
    "all_l1":     (1.0, 1.0, True,  []),
}


def find_latest_checkpoint(group, seed, base_dir, task="velocity"):
    rt, ct = GROUP_DIR_MAP[group]
    task_suffix = "" if task == "velocity" else f"_{task}"
    ckpt_dir = os.path.join(base_dir, f"ablation_{group}_{rt}_{ct}{task_suffix}",
                             f"seed_{seed}", "checkpoints")
    if not os.path.isdir(ckpt_dir):
        return ""
    return ckpt_dir


def make_env_config(curriculum_lambda, num_envs, ablation):
    cfg = TrainConfig()
    cfg.num_envs = num_envs
    cfg.num_steps = 128
    cfg.curriculum.lambda_start = curriculum_lambda
    cfg.ablation = ablation
    return cfg, cfg.get_runtime_env_config()


def override_phys_fields(phys_easy, phys_hard, fields):
    """Return a copy of phys_easy where listed fields are replaced from phys_hard."""
    if not fields:
        return phys_easy
    replacements = {f: getattr(phys_hard, f) for f in fields}
    return phys_easy.replace(**replacements)


def evaluate_dim(group, seed, dim, num_episodes, num_envs, base_dir,
                  task="velocity", rng_seed=12345):
    init_lambda, step_lambda, use_hard, fields = DIM_CONFIG[dim]

    ablation = AblationConfig.from_group(group)
    ablation.task = task
    cfg, env_cfg = make_env_config(init_lambda, num_envs, ablation)

    model, _, _ = create_stochastic_actor_critic(
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
    rng, rng_params, rng_reset, rng_loop = jax.random.split(rng, 4)

    # CRITICAL: use SAME RNG for both lambdas so that geometry/power/motor_max_thrust
    # are bit-identical between easy and hard. Only λ-controlled fields differ, so
    # per-dim overrides isolate the OOD effect without physical inconsistency
    # (e.g. heavy mass paired with weak motor_max_thrust).
    env_params_easy = generate_params(rng_params, num_envs, cfg.physics, 0.0)
    env_params_hard = generate_params(rng_params, num_envs, cfg.physics, 1.0)

    # Build per-dim params
    if use_hard:
        env_params = env_params_hard
    else:
        # Override selected fields on phys_params; replace into env_params_easy
        phys_easy = env_params_easy.phys_params
        phys_hard = env_params_hard.phys_params
        phys_new = override_phys_fields(phys_easy, phys_hard, fields)
        env_params = env_params_easy.replace(phys_params=phys_new)

    # Reset env with init_lambda
    state, _, obs = reset_env(rng_reset, num_envs, env_cfg, init_lambda, params=env_params)
    lstm_state = model.init_lstm_state(num_envs)

    # JIT-compiled step with step_lambda as runtime arg (so JIT compiles
    # once per group, not once per dim). step_lambda is small int → safe as
    # runtime numeric in JAX.
    step_lambda_arr = jnp.array(step_lambda, dtype=jnp.float32)

    @jax.jit
    def det_step(state, env_params, lstm_state, obs, rng_step, step_lam):
        action_mean, action_std, _, new_lstm = model.apply(params, obs, lstm_state, None)
        action = jnp.tanh(action_mean)
        new_state, timestep, new_env_params = step_env(
            rng_step, state, action, env_params, env_cfg, step_lam
        )
        done_mask = timestep.done[:, None].astype(jnp.float32)
        new_lstm = type(new_lstm)(
            h=new_lstm.h * (1.0 - done_mask),
            c=new_lstm.c * (1.0 - done_mask),
        )
        return new_state, new_env_params, new_lstm, timestep

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

    max_steps_per_episode = int(cfg.env.max_episode_time / 0.02)
    target_completed = num_episodes
    step_count = 0
    max_iter = max_steps_per_episode * 4

    while completed_episodes < target_completed and step_count < max_iter:
        rng_loop, step_rng = jax.random.split(rng_loop)
        state, env_params, lstm_state, timestep = det_step(
            state, env_params, lstm_state, obs, step_rng, step_lambda_arr
        )
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

        step_count += 1

    completed_returns = completed_returns[:target_completed]
    completed_lengths = completed_lengths[:target_completed]

    return {
        "group": group,
        "seed": seed,
        "dim": dim,
        "init_lambda": init_lambda,
        "step_lambda": step_lambda,
        "n_episodes": len(completed_returns),
        "mean_episode_return": float(np.mean(completed_returns)) if completed_returns else 0.0,
        "sd_episode_return":   float(np.std(completed_returns))  if completed_returns else 0.0,
        "mean_episode_length": float(np.mean(completed_lengths)) if completed_lengths else 0.0,
        "crash_rate":          crashed_count / max(target_completed, 1),
        "success_rate":        success_steps / max(total_steps, 1),
        "tracking_rmse":       float(np.sqrt(v_err_sq_sum / max(v_err_sq_n, 1))),
        "checkpoint_step":     int(step) if step is not None else 0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", nargs="+", default=["A", "B", "C", "D", "E", "F"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456, 789, 1024])
    ap.add_argument("--dims", nargs="+",
                    default=["nominal", "mass", "wind", "turb", "sensor", "actuator",
                             "init_state", "collision", "all_l1"],
                    choices=list(DIM_CONFIG.keys()))
    ap.add_argument("--num-episodes", type=int, default=100)
    ap.add_argument("--num-envs", type=int, default=256)
    ap.add_argument("--task", type=str, default="velocity")
    ap.add_argument("--base-dir", type=str,
                    default=str(Path(__file__).resolve().parents[3] / "experiments"))
    ap.add_argument("--output", type=str,
                    default=str(Path(__file__).resolve().parents[3] / "experiments" / "_h5_perdim" / "results.json"))
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    total = len(args.groups) * len(args.seeds) * len(args.dims)
    print(f"[*] H5 per-dim OOD eval — task={args.task}")
    print(f"    groups={args.groups}, seeds={args.seeds}, dims={args.dims}")
    print(f"    {args.num_episodes} episodes per (group, seed, dim) × {args.num_envs} envs")
    print(f"    Total evaluations: {total}")
    print()

    import traceback

    # Resume support: load existing results and skip done cells
    all_results = []
    done_keys = set()
    if os.path.exists(args.output):
        try:
            with open(args.output) as f:
                all_results = json.load(f)
            for r in all_results:
                if "error" not in r and "group" in r and "seed" in r and "dim" in r:
                    done_keys.add((r["group"], int(r["seed"]), r["dim"]))
            print(f"[*] RESUME: loaded {len(all_results)} records; "
                  f"{len(done_keys)} cells already done (skip)", flush=True)
        except Exception as e:
            print(f"[!] resume load failed ({e}); starting fresh", flush=True)
            all_results = []
            done_keys = set()

    t_start = time.time()
    done = 0
    for g in args.groups:
        for s in args.seeds:
            for dim in args.dims:
                if (g, int(s), dim) in done_keys:
                    done += 1
                    continue
                t1 = time.time()
                try:
                    r = evaluate_dim(g, s, dim, args.num_episodes, args.num_envs,
                                     args.base_dir, task=args.task)
                except Exception as e:
                    tb = traceback.format_exc()
                    r = {"group": g, "seed": s, "dim": dim, "error": str(e), "traceback": tb}
                    print(f"!!! EXCEPTION in G={g} seed={s} dim={dim}:\n{tb}", flush=True)
                done += 1
                dt = time.time() - t1
                eta = (time.time() - t_start) / done * (total - done) / 60.0
                err = f"  ERROR: {r.get('error', '')}" if "error" in r else ""
                print(f"[{done}/{total}] G={g} seed={s:>5d} dim={dim:<11s}  "
                      f"return={r.get('mean_episode_return', 0):.3f}  "
                      f"crash={r.get('crash_rate', 0):.2%}  "
                      f"({dt:.1f}s, ETA {eta:.1f} min){err}", flush=True)
                all_results.append(r)
                with open(args.output, "w") as f:
                    json.dump(all_results, f, indent=2)

    print(f"\n[+] Done in {(time.time()-t_start)/60:.1f} min. Results -> {args.output}", flush=True)


if __name__ == "__main__":
    main()
