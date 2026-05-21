"""
T3: Disturbance Rejection Task (v19.3).

Stateless disturbance schedule — episode_time-driven gusts and motor failures.
Tests robustness of the policy under in-flight perturbations.

Schedule (per episode, all times in seconds):
  t < 2.0s             : nominal flight (no disturbances)
  2.0 ≤ t < 4.0s       : strong wind gust (8 m/s, NED-x direction)
  4.0 ≤ t < 6.0s       : motor 0 efficiency drops to 50% (single motor degradation)
  6.0 ≤ t              : everything restored, observe recovery

The agent's REWARD and v_cmd computation are unchanged (use T1 sticks pipeline).
What changes is the EXPERIENCE during training: stronger physical disturbances
that the policy must learn to compensate.

Implementation: this module returns "override factors" that the env step applies
to the existing wind / motor params. Original PhysParams remain unchanged on disk;
the overrides apply only at the env-step call site.
"""
import jax.numpy as jnp
from jaxtyping import Array, Float
from typing import NamedTuple


# Disturbance schedule (s)
GUST_START = 2.0
GUST_END = 4.0
MOTOR_FAIL_START = 4.0
MOTOR_FAIL_END = 6.0

# Disturbance magnitudes — v19.3.1 调低后的默认值
# 早期版本曾用 (8 m/s, eff=0.5) 但太极端：8 m/s 风加上单电机减半几乎不可恢复，
# 学习信号过弱（agent 大概率全 episode 失控）。中等强度更适合学习+测试。
#
# 参考标尺:
#   - DR 训练中 wind_speed_max=15 m/s × λ=1.0；典型微风 1-3 m/s, 强阵风 5-8 m/s
#   - 4 m/s 阵风 ≈ 中等强度（飞行器明显受扰但不必然失控）
#   - motor_eff=0.7 (即 30% loss) ≈ 单电机轻度老化（明显能感知，可以补偿）
GUST_WIND_SPEED_MS = 4.0        # NED-x gust speed (m/s) — was 8.0
MOTOR_EFF_DEGRADED = 0.7        # motor 0 efficiency factor — was 0.5


class DisturbanceOverride(NamedTuple):
    """Static overrides for env step under T3 disturbance task.

    All fields shape [batch] (or [batch, 4] for motor_efficiency).
    """
    wind_speed_override: Float[Array, "batch"]            # m/s; replaces phys_params.wind_speed_mean
    wind_direction_override: Float[Array, "batch"]        # rad; replaces phys_params.wind_direction
    apply_wind_override: Float[Array, "batch"]            # 0/1 mask (apply only during gust window)
    motor_efficiency_factor: Float[Array, "batch 4"]      # multiplicative on phys_params.motor_efficiency


def compute_disturbance_overrides(
    episode_time: Float[Array, "batch"],
) -> DisturbanceOverride:
    """Compute episode_time-driven disturbance overrides.

    Args:
        episode_time: [batch] elapsed time in episode (s)

    Returns:
        DisturbanceOverride struct with per-env override values.
    """
    # Wind gust window: GUST_START ≤ t < GUST_END
    in_gust = (episode_time >= GUST_START) & (episode_time < GUST_END)
    wind_speed = jnp.where(in_gust, GUST_WIND_SPEED_MS, 0.0)
    wind_direction = jnp.zeros_like(episode_time)              # NED-x direction (0 rad)
    apply_wind = in_gust.astype(jnp.float32)

    # Motor 0 failure window: MOTOR_FAIL_START ≤ t < MOTOR_FAIL_END
    in_motor_fail = (episode_time >= MOTOR_FAIL_START) & (episode_time < MOTOR_FAIL_END)
    # When in failure window, motor 0 efficiency factor = 0.5; else 1.0
    motor_0_factor = jnp.where(in_motor_fail, MOTOR_EFF_DEGRADED, 1.0)
    motor_factors = jnp.stack([
        motor_0_factor,
        jnp.ones_like(episode_time),
        jnp.ones_like(episode_time),
        jnp.ones_like(episode_time),
    ], axis=-1)                                                # [batch, 4]

    return DisturbanceOverride(
        wind_speed_override=wind_speed,
        wind_direction_override=wind_direction,
        apply_wind_override=apply_wind,
        motor_efficiency_factor=motor_factors,
    )
