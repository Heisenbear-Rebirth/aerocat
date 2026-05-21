"""
v19.3 Multi-task module for AeroCat ablation experiments.

Tasks (mutually exclusive, selected via AblationConfig.task / EnvConfig.task):

  - "velocity"    (T1, default)  — random sticks → l1_process → v_cmd
                                   This is the v18-original task; all v19.0/v19.2
                                   experiments (groups A-F) use this task.

  - "waypoint"    (T2)           — figure-8 pattern in xy-plane, P-controller
                                   produces v_cmd from current position vs target.
                                   STATELESS — uses episode_time only, no extra
                                   state in EnvState. Tests generalization to
                                   smooth deployment-style velocity profiles.

  - "disturbance" (T3)           — same sticks as T1, but with episode_time-
                                   scheduled wind gusts and motor failures.
                                   STATELESS — disturbance schedule purely a
                                   function of episode_time. Tests robustness.

Design principle: ADDITIVE — none of the new tasks touch EnvState/PhysState
structure. Saved checkpoints from v19.0/v19.2 (groups A-F) remain loadable.
"""
from .waypoint import compute_waypoint_v_cmd_heading
from .disturbance import compute_disturbance_overrides

__all__ = [
    "compute_waypoint_v_cmd_heading",
    "compute_disturbance_overrides",
]
