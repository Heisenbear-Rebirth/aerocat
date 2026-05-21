"""
T2: Waypoint Navigation Task (v19.3).

Stateless waypoint generator — uses episode_time only, no extra state needed.
Pattern: figure-8 in xy-plane at constant altitude (NED frame).

Mathematical pattern (NED, units m and s):
  x_target(t) = A · sin(ω·t)
  y_target(t) = (A/2) · sin(2ω·t)        (figure-8: y has 2× freq)
  z_target(t) = z₀                       (constant altitude, NED z negative = up)

  ω = 2π / T,  T = 5.0s,  A = 2.5m,  z₀ = -2.0m  (2m above ground)

The agent's task is unchanged from T1's perspective: track v_cmd_heading.
Only the SOURCE of v_cmd is different — here it's a P-controller on position
error rather than OU-driven random sticks.

This task tests generalization: if PSC's structural prior is robust, it should
also work on this smoother velocity profile (typical of real-world waypoint
following). v_cmd from P-controller decays smoothly as drone approaches target,
unlike T1's discontinuous joystick-style commands.
"""
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from ..core.state import PhysState


# Default waypoint parameters
DEFAULT_AMPLITUDE = 2.5         # figure-8 horizontal radius (m)
DEFAULT_PERIOD = 5.0            # full figure-8 period (s)
DEFAULT_ALTITUDE = -2.0         # NED z (negative = up): 2m above ground
DEFAULT_KP = 1.5                # P-controller gain (m/s per m)
DEFAULT_V_MAX = 5.0             # velocity command saturation (m/s)


def _quaternion_to_yaw(q: Float[Array, "batch 4"]) -> Float[Array, "batch"]:
    """Extract yaw (NED z-axis rotation) from quaternion [w, x, y, z]."""
    qw = q[..., 0]
    qx = q[..., 1]
    qy = q[..., 2]
    qz = q[..., 3]
    return jnp.arctan2(
        2.0 * (qw * qz + qx * qy),
        1.0 - 2.0 * (qy * qy + qz * qz),
    )


def compute_waypoint_position(
    episode_time: Float[Array, "batch"],
    amplitude: float = DEFAULT_AMPLITUDE,
    period: float = DEFAULT_PERIOD,
    altitude: float = DEFAULT_ALTITUDE,
) -> Float[Array, "batch 3"]:
    """Stateless figure-8 waypoint position in NED frame.

    Returns target [x, y, z]_NED at time t = episode_time.
    """
    omega = 2.0 * jnp.pi / period
    x = amplitude * jnp.sin(omega * episode_time)
    y = 0.5 * amplitude * jnp.sin(2.0 * omega * episode_time)
    z = jnp.full_like(episode_time, altitude)
    return jnp.stack([x, y, z], axis=-1)   # [batch, 3]


def compute_waypoint_v_cmd_heading(
    phys_state: PhysState,
    episode_time: Float[Array, "batch"],
    amplitude: float = DEFAULT_AMPLITUDE,
    period: float = DEFAULT_PERIOD,
    altitude: float = DEFAULT_ALTITUDE,
    kp: float = DEFAULT_KP,
    v_max: float = DEFAULT_V_MAX,
) -> Float[Array, "batch 3"]:
    """Compute v_cmd in heading frame from waypoint position error (P-controller).

    Pipeline:
      1. Compute target_NED(t) — figure-8 position in NED frame.
      2. pos_error_NED = target_NED - current_position_NED
      3. Project pos_error to heading frame using current yaw.
      4. v_cmd_heading = kp · pos_error_heading, clipped to v_max norm.

    Returns:
      v_cmd_heading: [batch, 3] velocity command in heading frame (forward, right, down).
    """
    # 1. Target position in NED
    target_ned = compute_waypoint_position(episode_time, amplitude, period, altitude)

    # 2. Position error in NED (target - current)
    pos_error_ned = target_ned - phys_state.position    # [batch, 3]

    # 3. NED → heading frame (rotate by current yaw)
    yaw = _quaternion_to_yaw(phys_state.quaternion)     # [batch]
    cos_y = jnp.cos(yaw)
    sin_y = jnp.sin(yaw)
    err_fwd = cos_y * pos_error_ned[..., 0] + sin_y * pos_error_ned[..., 1]
    err_right = -sin_y * pos_error_ned[..., 0] + cos_y * pos_error_ned[..., 1]
    err_down = pos_error_ned[..., 2]                    # heading-frame z = NED z
    pos_error_heading = jnp.stack([err_fwd, err_right, err_down], axis=-1)

    # 4. P-controller with norm clipping
    v_cmd = kp * pos_error_heading                       # [batch, 3]
    v_norm = jnp.sqrt(jnp.sum(v_cmd ** 2, axis=-1, keepdims=True) + 1e-8)
    scale = jnp.minimum(1.0, v_max / v_norm)
    v_cmd = v_cmd * scale

    return v_cmd
