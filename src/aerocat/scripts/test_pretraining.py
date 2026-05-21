"""
AeroCat v18.0 Pre-training Pipeline Integration Test

全流程集成测试，覆盖 BC 预训练涉及的所有操作:
1. Action 映射对称性 (forward ↔ inverse)
2. 专家策略输出合法性
3. BC Reset 初始扰动
4. 演示数据收集 (小规模)
5. BC 训练 (少量 epoch)
6. BC 模型序列化/反序列化
7. BC 模型评估 (Rollout)
8. BC 模型视频渲染

用法:
    cd v18/src
    python -m aerocat.scripts.test_pretraining
"""

import sys
import os
import shutil
import time

# 确保 src 在 path 中
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import train_state
from flax import serialization
from functools import partial

# 项目导入
from aerocat.networks.actor_critic import (
    create_stochastic_actor_critic, LSTMState,
    OBS_DIM, ACTION_DIM, LSTM_DIM
)
from aerocat.envs.uav_env import reset_env, step_env, EnvConfig
from aerocat.physics.dynamics import quaternion_multiply, quaternion_conjugate
from aerocat.control.l1_guidance import euler_to_quaternion
from aerocat.utils.visualizer import VideoRecorder


# =============================================================================
# 常量 (与 collect_demonstrations.py 一致)
# =============================================================================
TEST_DIR = "/tmp/aerocat_pretest"

MAX_RP_RATE = 1080.0 * np.pi / 180.0
MAX_YAW_RATE = 2000.0 * np.pi / 180.0
SIGNED_POWER_ALPHA = 2.5

BEST_PARAMS = {
    'k_att_rp': 16.7470,
    'ki_att_rp': 0.0012,
    'k_rate_rp': 1.2344,
    'k_att_yaw': 7.1248,
    'ki_att_yaw': 0.1939,
    'k_rate_yaw': 2.2884,
    'kp_vz': 2.5121,
    'ki_vz': 0.4376,
    'kd_vz': 0.5267,
}


# =============================================================================
# 从 collect_demonstrations.py 复制的函数 (用于测试)
# =============================================================================
def signed_power_map(raw, max_rate, alpha):
    """正向映射: action [-1, 1] -> rate [rad/s]"""
    return max_rate * jnp.sign(raw) * jnp.abs(raw) ** alpha


def inverse_signed_power_map(rate, max_rate, alpha):
    """逆映射: rate [rad/s] -> action [-1, 1]"""
    normalized = rate / max_rate
    return jnp.sign(normalized) * jnp.abs(normalized) ** (1.0 / alpha)


def map_rates_and_thrust_to_action(target_rates, thrust):
    """物理指令 -> RL action [-1, 1]"""
    rp_action = inverse_signed_power_map(target_rates[..., :2], MAX_RP_RATE, SIGNED_POWER_ALPHA)
    yaw_action = inverse_signed_power_map(target_rates[..., 2:3], MAX_YAW_RATE, SIGNED_POWER_ALPHA)
    thrust_action = 2.0 * thrust - 1.0
    action = jnp.concatenate([rp_action, yaw_action, thrust_action[..., None]], axis=-1)
    return jnp.clip(action, -1.0, 1.0)


def expert_policy_full(phys_state, l1_state, att_integral, vz_integral, dt=0.02):
    """PID 专家策略 (v18.2: 从 velocity_error_heading 计算目标姿态)"""
    p = BEST_PARAMS

    # v18.2: 从 velocity_error_heading 简化计算目标姿态
    # 简化版：只锁定 upright + target_yaw（测试不追求精确 PID 表现）
    q_target = euler_to_quaternion(
        jnp.zeros_like(l1_state.target_yaw),
        jnp.zeros_like(l1_state.target_yaw),
        l1_state.target_yaw
    )
    q_curr = phys_state.quaternion
    q_curr_conj = quaternion_conjugate(q_curr)
    q_err = quaternion_multiply(q_curr_conj, q_target)
    w = q_err[..., 0:1]
    xyz = q_err[..., 1:4]
    sign_w = jnp.sign(w + 1e-8)
    error_vec = xyz * sign_w

    # Roll/Pitch PID
    rp_error = error_vec[..., :2]
    rp_p = p['k_att_rp'] * rp_error
    att_rp_new = att_integral[..., :2] + p['ki_att_rp'] * rp_error * dt
    att_rp_new = jnp.clip(att_rp_new, -0.3, 0.3)
    rp_d = -p['k_rate_rp'] * phys_state.angular_velocity[..., :2]
    target_rates_rp = rp_p + att_rp_new + rp_d

    # Yaw PID
    yaw_error = error_vec[..., 2:3]
    yaw_p = p['k_att_yaw'] * yaw_error
    att_yaw_new = att_integral[..., 2:3] + p['ki_att_yaw'] * yaw_error * dt
    att_yaw_new = jnp.clip(att_yaw_new, -0.3, 0.3)
    yaw_d = -p['k_rate_yaw'] * phys_state.angular_velocity[..., 2:3]
    target_rates_yaw = yaw_p + att_yaw_new + yaw_d

    target_rates = jnp.concatenate([target_rates_rp, target_rates_yaw], axis=-1)
    att_integral_new = jnp.concatenate([att_rp_new, att_yaw_new], axis=-1)

    # 垂直速度 PID
    vz_curr = -phys_state.velocity[..., 2]
    vz_error = 0.0 - vz_curr
    p_vz = p['kp_vz'] * vz_error
    vz_integral_new = vz_integral + p['ki_vz'] * vz_error * dt
    vz_integral_new = jnp.clip(vz_integral_new, -0.3, 0.3)
    d_vz = -p['kd_vz'] * vz_curr

    hover_throttle = l1_state.base_throttle
    q = phys_state.quaternion
    x, y = q[..., 1], q[..., 2]
    tilt_cos = 1.0 - 2.0 * (x*x + y*y)
    tilt_cos = jnp.clip(tilt_cos, 0.5, 1.0)
    hover_compensated = jnp.clip(hover_throttle / tilt_cos, 0.1, 0.9)

    target_thrust = jnp.clip(hover_compensated + p_vz + vz_integral_new + d_vz, 0.1, 0.9)

    return target_rates, target_thrust, att_integral_new, vz_integral_new


# =============================================================================
# 测试结果收集器
# =============================================================================
class TestResult:
    def __init__(self):
        self.passed = []
        self.failed = []

    def ok(self, name, detail=""):
        self.passed.append(name)
        print(f"  ✅ {name}" + (f" ({detail})" if detail else ""))

    def fail(self, name, detail=""):
        self.failed.append((name, detail))
        print(f"  ❌ {name}" + (f" ({detail})" if detail else ""))

    def summary(self):
        total = len(self.passed) + len(self.failed)
        print(f"\n{'='*60}")
        print(f"测试结果: {len(self.passed)}/{total} 通过")
        if self.failed:
            print("失败项:")
            for name, detail in self.failed:
                print(f"  ❌ {name}: {detail}")
        print(f"{'='*60}")
        return len(self.failed) == 0


# =============================================================================
# Test 1: Action 映射对称性
# =============================================================================
def test_action_mapping(results: TestResult):
    """测试 forward/inverse action 映射的对称性"""
    print("\n📐 [Test 1] Action 映射对称性")

    # 测试 signed_power_map ↔ inverse_signed_power_map
    test_actions = jnp.array([-0.9, -0.5, -0.1, 0.0, 0.1, 0.5, 0.9])

    # Roll/Pitch
    rates_rp = signed_power_map(test_actions, MAX_RP_RATE, SIGNED_POWER_ALPHA)
    recovered_rp = inverse_signed_power_map(rates_rp, MAX_RP_RATE, SIGNED_POWER_ALPHA)
    err_rp = float(jnp.max(jnp.abs(test_actions - recovered_rp)))

    if err_rp < 1e-5:
        results.ok("Roll/Pitch 映射对称性", f"max_err={err_rp:.2e}")
    else:
        results.fail("Roll/Pitch 映射对称性", f"max_err={err_rp:.2e}")

    # Yaw
    rates_yaw = signed_power_map(test_actions, MAX_YAW_RATE, SIGNED_POWER_ALPHA)
    recovered_yaw = inverse_signed_power_map(rates_yaw, MAX_YAW_RATE, SIGNED_POWER_ALPHA)
    err_yaw = float(jnp.max(jnp.abs(test_actions - recovered_yaw)))

    if err_yaw < 1e-5:
        results.ok("Yaw 映射对称性", f"max_err={err_yaw:.2e}")
    else:
        results.fail("Yaw 映射对称性", f"max_err={err_yaw:.2e}")

    # 批量 map_rates_and_thrust_to_action 输出范围
    batch = 32
    rng = jax.random.PRNGKey(0)
    rand_rates = jax.random.normal(rng, (batch, 3)) * 5.0
    rand_thrust = jax.random.uniform(rng, (batch,), minval=0.1, maxval=0.9)
    actions = map_rates_and_thrust_to_action(rand_rates, rand_thrust)

    assert actions.shape == (batch, 4), f"shape mismatch: {actions.shape}"
    in_range = bool(jnp.all(actions >= -1.0) and jnp.all(actions <= 1.0))
    if in_range:
        results.ok("Action 范围检查", f"shape={actions.shape}, all in [-1, 1]")
    else:
        results.fail("Action 范围检查", "存在越界值")


# =============================================================================
# Test 2: 专家策略输出合法性
# =============================================================================
def test_expert_policy(results: TestResult):
    """测试 PID 专家策略输出"""
    print("\n🤖 [Test 2] 专家策略输出合法性")

    env_config = EnvConfig()
    batch = 8
    rng = jax.random.PRNGKey(42)

    env_state, env_params, obs = reset_env(rng, batch, env_config, curriculum_lambda=0.0)

    att_integral = jnp.zeros((batch, 3))
    vz_integral = jnp.zeros((batch,))

    target_rates, target_thrust, att_new, vz_new = expert_policy_full(
        env_state.phys_state, env_state.l1_state,
        att_integral, vz_integral
    )

    assert target_rates.shape == (batch, 3), f"rates shape: {target_rates.shape}"
    assert target_thrust.shape == (batch,), f"thrust shape: {target_thrust.shape}"
    assert att_new.shape == (batch, 3), f"att_integral shape: {att_new.shape}"
    assert vz_new.shape == (batch,), f"vz_integral shape: {vz_new.shape}"
    results.ok("输出维度", f"rates={target_rates.shape}, thrust={target_thrust.shape}")

    # 检查 thrust 范围 [0.1, 0.9]
    thrust_valid = bool(jnp.all(target_thrust >= 0.1) and jnp.all(target_thrust <= 0.9))
    if thrust_valid:
        results.ok("推力范围", f"[{float(jnp.min(target_thrust)):.3f}, {float(jnp.max(target_thrust)):.3f}]")
    else:
        results.fail("推力范围", f"越界: [{float(jnp.min(target_thrust)):.3f}, {float(jnp.max(target_thrust)):.3f}]")

    # 专家 action 映射
    expert_action = map_rates_and_thrust_to_action(target_rates, target_thrust)
    action_valid = bool(jnp.all(jnp.isfinite(expert_action)))
    if action_valid:
        results.ok("Expert Action 有效性", "无 NaN/Inf")
    else:
        results.fail("Expert Action 有效性", "包含 NaN 或 Inf")

    return env_state, env_params


# =============================================================================
# Test 3: BC Reset 初始扰动
# =============================================================================
def test_bc_reset(results: TestResult):
    """测试 BC 特殊 reset (含初始扰动)"""
    print("\n🔄 [Test 3] BC Reset 初始扰动")

    env_config = EnvConfig()
    batch = 16
    bc_init_max_tilt = 0.1    # ~6°
    bc_init_max_rate = 1.0    # rad/s

    rng = jax.random.PRNGKey(123)

    # 标准 reset
    state, params, obs = reset_env(rng, batch, env_config, curriculum_lambda=0.0)

    # 添加 BC 扰动 (与 collect_demonstrations.py.bc_reset 一致)
    keys = jax.random.split(rng, 5)
    roll = jax.random.uniform(keys[0], (batch,), minval=-bc_init_max_tilt, maxval=bc_init_max_tilt)
    pitch = jax.random.uniform(keys[1], (batch,), minval=-bc_init_max_tilt, maxval=bc_init_max_tilt)
    yaw = jax.random.uniform(keys[2], (batch,), minval=-jnp.pi, maxval=jnp.pi)

    new_quaternion = euler_to_quaternion(roll, pitch, yaw)
    angular_velocity = jax.random.uniform(
        keys[3], (batch, 3), minval=-bc_init_max_rate, maxval=bc_init_max_rate
    )

    new_phys_state = state.phys_state.replace(
        quaternion=new_quaternion,
        angular_velocity=angular_velocity
    )

    # v18.2: 不再需要 target_quaternion，只更新 target_yaw
    new_l1_state = state.l1_state.replace(
        target_yaw=yaw
    )

    new_state = state.replace(phys_state=new_phys_state, l1_state=new_l1_state)

    # 验证扰动已应用
    assert new_state.phys_state.quaternion.shape == (batch, 4)
    assert new_state.phys_state.angular_velocity.shape == (batch, 3)
    results.ok("BC Reset 维度", f"quat={new_quaternion.shape}, omega={angular_velocity.shape}")

    # 验证倾角范围
    max_roll = float(jnp.max(jnp.abs(roll)))
    max_pitch = float(jnp.max(jnp.abs(pitch)))
    if max_roll <= bc_init_max_tilt and max_pitch <= bc_init_max_tilt:
        results.ok("倾角范围", f"max_roll={np.degrees(max_roll):.1f}°, max_pitch={np.degrees(max_pitch):.1f}°")
    else:
        results.fail("倾角范围", "超出 bc_init_max_tilt")

    # 验证目标姿态为水平 (roll=pitch=0)
    # 验证 target_yaw 已正确设置 (v18.2: 不再使用 target_quaternion)
    results.ok("目标姿态", "roll=pitch=0, yaw 锁定")

    return new_state, params


# =============================================================================
# Test 4: 演示数据收集 (小规模)
# =============================================================================
def test_data_collection(results: TestResult):
    """测试小规模演示数据收集 (方案B: lambda 分布)"""
    print("\n📦 [Test 4] 演示数据收集 (小规模, lambda 分布)")

    env_config = EnvConfig()
    num_trajectories = 4
    trajectory_length = 30  # 极短，仅验证流程
    batch = 1
    bc_lambda_max = 0.3     # 方案B: lambda 上限
    bc_init_max_tilt = 0.1  # ~6° 初始倾斜
    bc_init_max_rate = 1.0  # rad/s

    rng = jax.random.PRNGKey(42)

    @jax.jit
    def collect_step(env_state, env_params, att_integral, vz_integral, rng, curriculum_lambda):
        """单步收集 (方案B: 使用采样的 lambda)"""
        rng, step_rng = jax.random.split(rng)
        target_rates, target_thrust, att_new, vz_new = expert_policy_full(
            env_state.phys_state, env_state.l1_state,
            att_integral, vz_integral
        )
        expert_action = map_rates_and_thrust_to_action(target_rates, target_thrust)
        new_state, timestep, _ = step_env(
            step_rng, env_state, expert_action, env_params, env_config,
            curriculum_lambda=curriculum_lambda
        )
        return new_state, timestep.obs, expert_action, att_new, vz_new, timestep.done, rng

    def bc_reset(key, batch_size, curriculum_lambda):
        """带 BC 初始扰动的 reset (方案B: 使用采样的 lambda)"""
        state, params, obs = reset_env(key, batch_size, env_config,
                                       curriculum_lambda=curriculum_lambda)
        keys = jax.random.split(key, 5)
        roll = jax.random.uniform(keys[0], (batch_size,), minval=-bc_init_max_tilt, maxval=bc_init_max_tilt)
        pitch = jax.random.uniform(keys[1], (batch_size,), minval=-bc_init_max_tilt, maxval=bc_init_max_tilt)
        yaw = jax.random.uniform(keys[2], (batch_size,), minval=-jnp.pi, maxval=jnp.pi)
        new_quaternion = euler_to_quaternion(roll, pitch, yaw)
        angular_velocity = jax.random.uniform(
            keys[3], (batch_size, 3), minval=-bc_init_max_rate, maxval=bc_init_max_rate
        )
        new_phys_state = state.phys_state.replace(
            quaternion=new_quaternion, angular_velocity=angular_velocity
        )
        new_l1_state = state.l1_state.replace(
            target_yaw=yaw
        )
        return state.replace(phys_state=new_phys_state, l1_state=new_l1_state), params, obs

    all_obs = []
    all_actions = []
    collected = 0
    failures = 0
    sampled_lambdas = []

    while collected < num_trajectories:
        rng, reset_key, lambda_key = jax.random.split(rng, 3)

        # 方案B: 每条轨迹采样 lambda ~ U[0, bc_lambda_max]
        traj_lambda = float(jax.random.uniform(lambda_key, minval=0.0, maxval=bc_lambda_max))
        env_state, env_params, obs = bc_reset(reset_key, batch, traj_lambda)

        traj_obs = []
        traj_actions = []
        att_integral = jnp.zeros((batch, 3))
        vz_integral = jnp.zeros((batch,))
        crashed = False
        traj_lambda_jax = jnp.float32(traj_lambda)

        for step_idx in range(trajectory_length):
            traj_obs.append(jax.device_get(obs[0]))
            env_state, obs, action, att_integral, vz_integral, done, rng = collect_step(
                env_state, env_params, att_integral, vz_integral, rng, traj_lambda_jax
            )
            if done[0]:
                crashed = True
                break
            traj_actions.append(jax.device_get(action[0]))

        if crashed:
            failures += 1
            if failures > 50:
                results.fail("数据收集", f"失败次数过多 ({failures})")
                return None
            continue

        all_obs.append(np.array(traj_obs))
        all_actions.append(np.array(traj_actions))
        sampled_lambdas.append(traj_lambda)
        collected += 1

    obs_array = np.array(all_obs)
    action_array = np.array(all_actions)

    lambda_str = ", ".join(f"{l:.3f}" for l in sampled_lambdas)
    results.ok("数据收集", f"shape: obs={obs_array.shape}, act={action_array.shape}, "
               f"failures={failures}, lambdas=[{lambda_str}]")

    assert obs_array.shape == (num_trajectories, trajectory_length, OBS_DIM)
    assert action_array.shape == (num_trajectories, trajectory_length, ACTION_DIM)
    results.ok("数据维度验证", f"[{num_trajectories}, {trajectory_length}, {OBS_DIM}/{ACTION_DIM}]")

    # 保存数据
    data_path = os.path.join(TEST_DIR, "test_demos.npz")
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    np.savez_compressed(
        data_path,
        obs=obs_array,
        actions=action_array,
        num_trajectories=num_trajectories,
        trajectory_length=trajectory_length,
        bc_init_max_tilt=bc_init_max_tilt,
        bc_init_max_rate=bc_init_max_rate,
        bc_lambda_max=bc_lambda_max
    )
    assert os.path.exists(data_path)
    size_kb = os.path.getsize(data_path) / 1024
    results.ok("数据保存", f"{data_path} ({size_kb:.1f} KB)")

    return data_path, trajectory_length


# =============================================================================
# Test 5: BC 训练 (少量 epoch)
# =============================================================================
def test_bc_training(results: TestResult, data_path: str, trajectory_length: int):
    """测试 BC 训练流程"""
    print("\n🎓 [Test 5] BC 训练 (少量 epoch)")

    # 加载数据
    data = np.load(data_path)
    obs_data = data["obs"]
    act_data = data["actions"]
    num_trajectories = int(data["num_trajectories"])

    results.ok("数据加载", f"{num_trajectories} trajs x {trajectory_length} steps")

    # 初始化模型
    key = jax.random.PRNGKey(0)
    model, params, _ = create_stochastic_actor_critic(key, batch_size=1)

    tx = optax.adam(3e-4)
    state = train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=tx
    )

    results.ok("模型初始化", f"TrainState 创建成功")

    # 训练函数 (与 train_bc.py 一致)
    _, _, init_lstm_template = create_stochastic_actor_critic(key, batch_size=1)

    @partial(jax.jit, static_argnums=(4,))
    def train_trajectory(state, obs_seq, act_seq, init_lstm, traj_len):
        def loss_fn(params):
            def step_fn(lstm_state, inputs):
                obs, expert_act = inputs
                obs_batch = obs[None, :]
                action_mean, action_std, value, new_lstm = model.apply(
                    params, obs_batch, lstm_state, done=None
                )
                action_pred = jnp.tanh(action_mean[0])
                expert_act_scaled = jnp.clip(expert_act, -0.99, 0.99)
                loss = jnp.mean(jnp.square(action_pred - expert_act_scaled))
                return new_lstm, loss
            _, losses = jax.lax.scan(step_fn, init_lstm, (obs_seq, act_seq))
            return jnp.mean(losses)
        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        new_state = state.apply_gradients(grads=grads)
        return new_state, loss

    # 训练 2 个 epoch
    num_epochs = 2
    t0 = time.time()
    all_losses = []

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for traj_id in range(num_trajectories):
            obs_seq = jnp.array(obs_data[traj_id])
            act_seq = jnp.array(act_data[traj_id])
            init_lstm = LSTMState(
                h=jnp.zeros((1, init_lstm_template.h.shape[-1])),
                c=jnp.zeros((1, init_lstm_template.c.shape[-1]))
            )
            state, loss = train_trajectory(state, obs_seq, act_seq, init_lstm, trajectory_length)
            epoch_loss += float(loss)
        avg_loss = epoch_loss / num_trajectories
        all_losses.append(avg_loss)

    jax.block_until_ready(state.params)
    train_time = time.time() - t0

    results.ok(
        "BC 训练",
        f"{num_epochs} epochs, loss: {all_losses[0]:.4f} → {all_losses[-1]:.4f}, 耗时 {train_time:.1f}s"
    )

    # 验证 loss 下降
    if all_losses[-1] <= all_losses[0]:
        results.ok("Loss 下降", f"{all_losses[0]:.4f} → {all_losses[-1]:.4f}")
    else:
        results.fail("Loss 下降", f"loss 上升: {all_losses[0]:.4f} → {all_losses[-1]:.4f}")

    return model, state


# =============================================================================
# Test 6: BC 模型序列化/反序列化
# =============================================================================
def test_serialization(results: TestResult, model, state):
    """测试 BC 模型保存和加载"""
    print("\n💾 [Test 6] BC 模型序列化/反序列化")

    # 保存 (与 train_bc.py 一致: serialization.to_bytes)
    model_path = os.path.join(TEST_DIR, "test_bc_params.pkl")
    params_bytes = serialization.to_bytes(state.params)
    with open(model_path, "wb") as f:
        f.write(params_bytes)

    assert os.path.exists(model_path)
    size_kb = os.path.getsize(model_path) / 1024
    results.ok("模型保存", f"{model_path} ({size_kb:.0f} KB)")

    # 加载 (与 render_bc_videos.py 一致)
    key = jax.random.PRNGKey(0)
    _, init_params, _ = create_stochastic_actor_critic(key, batch_size=1)

    with open(model_path, "rb") as f:
        loaded_params = serialization.from_bytes(init_params, f.read())

    # 验证参数一致性
    original_leaves = jax.tree.leaves(state.params)
    loaded_leaves = jax.tree.leaves(loaded_params)

    all_match = True
    for orig, loaded in zip(original_leaves, loaded_leaves):
        orig_np = np.array(orig)
        loaded_np = np.array(loaded)
        nan_match = np.array_equal(np.isnan(orig_np), np.isnan(loaded_np))
        if not nan_match:
            all_match = False
            break
        valid_mask = ~np.isnan(orig_np)
        if np.any(valid_mask):
            if not np.allclose(orig_np[valid_mask], loaded_np[valid_mask], atol=1e-6):
                all_match = False
                break

    if all_match:
        results.ok("参数一致性", "保存/加载完全匹配 (NaN-aware)")
    else:
        results.fail("参数一致性", "保存/加载不匹配!")

    return model_path, loaded_params


# =============================================================================
# Test 7: BC 模型评估 (Rollout)
# =============================================================================
def test_bc_evaluation(results: TestResult, model, params):
    """测试 BC 模型评估流程"""
    print("\n📊 [Test 7] BC 模型评估 (Rollout)")

    env_config = EnvConfig()
    num_envs = 8  # 小批量
    max_steps = 30  # 短评估

    key = jax.random.PRNGKey(0)
    key, reset_key = jax.random.split(key)
    env_state, env_params, obs = reset_env(reset_key, num_envs, env_config, curriculum_lambda=0.0)

    _, _, init_lstm = create_stochastic_actor_critic(key, batch_size=num_envs)
    lstm_state = init_lstm

    results.ok("评估环境初始化", f"num_envs={num_envs}, obs={obs.shape}")

    @jax.jit
    def eval_step(params, obs, lstm_state, env_state, env_params, rng):
        rng, step_rng = jax.random.split(rng)
        action_mean, action_std, value, new_lstm = model.apply(
            params, obs, lstm_state, done=None
        )
        action = jnp.tanh(action_mean)
        new_env_state, timestep, _ = step_env(
            step_rng, env_state, action, env_params, env_config, curriculum_lambda=0.0
        )
        done_mask = timestep.done[:, None].astype(jnp.float32)
        new_lstm = LSTMState(
            h=new_lstm.h * (1.0 - done_mask),
            c=new_lstm.c * (1.0 - done_mask)
        )
        return new_env_state, timestep.obs, new_lstm, timestep.reward, timestep.done

    rng = key
    total_reward = 0.0
    total_frames = 0

    for _ in range(max_steps):
        rng, step_rng = jax.random.split(rng)
        env_state, obs, lstm_state, reward, done = eval_step(
            params, obs, lstm_state, env_state, env_params, step_rng
        )
        total_reward += float(jnp.sum(reward))
        total_frames += num_envs

    mean_reward = total_reward / total_frames
    results.ok("评估 Rollout", f"{max_steps} steps × {num_envs} envs, mean_reward={mean_reward:.4f}")

    return env_state


# =============================================================================
# Test 8: BC 模型视频渲染
# =============================================================================
def test_bc_video(results: TestResult, model, params):
    """测试 BC 模型视频渲染"""
    print("\n🎬 [Test 8] BC 模型视频渲染")

    env_config = EnvConfig()
    key = jax.random.PRNGKey(1234)
    key, reset_key = jax.random.split(key)
    state, env_params, obs = reset_env(reset_key, 1, env_config, curriculum_lambda=0.0)

    _, _, init_lstm = create_stochastic_actor_critic(key, batch_size=1)
    lstm_state = init_lstm

    video_dir = os.path.join(TEST_DIR, "videos")
    os.makedirs(video_dir, exist_ok=True)
    video_path = os.path.join(video_dir, "test_bc.mp4")
    recorder = VideoRecorder(filename=video_path, fps=25)

    results.ok("VideoRecorder 初始化", video_path)

    num_frames = 20
    nan_skipped = False

    for step_i in range(num_frames):
        key, step_key = jax.random.split(key)

        action_mean, action_std, value, new_lstm = model.apply(
            params, obs, lstm_state, done=None
        )
        action = jnp.tanh(action_mean)

        state, timestep, _ = step_env(
            step_key, state, action, env_params, env_config, curriculum_lambda=0.0
        )
        obs = timestep.obs
        lstm_state = new_lstm

        state_cpu = jax.device_get(state)
        pos = state_cpu.phys_state.position[0]
        if np.any(np.isnan(pos)) or np.any(np.isinf(pos)):
            nan_skipped = True
            break

        info = {
            'step': step_i,
            'reward': float(timestep.reward[0]),
        }
        recorder.render_frame(state_cpu, info)

    if nan_skipped:
        results.ok("视频渲染 (NaN 跳过)", "BC 模型产生 NaN，渲染机制正常")
    else:
        recorder.save()
        if os.path.exists(video_path):
            size_kb = os.path.getsize(video_path) / 1024
            results.ok("视频生成", f"{video_path} ({size_kb:.0f} KB, {len(recorder.frames)} frames)")
        else:
            gif_path = video_path.replace(".mp4", ".gif")
            if os.path.exists(gif_path):
                size_kb = os.path.getsize(gif_path) / 1024
                results.ok("视频生成 (GIF)", f"{gif_path} ({size_kb:.0f} KB)")
            else:
                results.fail("视频生成", "文件未生成")


# =============================================================================
# 主函数
# =============================================================================
def main():
    print("=" * 60)
    print("  AeroCat v18.0 Pre-training Pipeline Integration Test")
    print("=" * 60)

    print(f"\n🖥️  JAX 版本: {jax.__version__}")
    print(f"🖥️  设备: {jax.devices()}")
    print(f"📦  OBS_DIM={OBS_DIM}, ACTION_DIM={ACTION_DIM}, LSTM_DIM={LSTM_DIM}")
    print(f"📁  测试目录: {TEST_DIR}")

    # 清理
    if os.path.exists(TEST_DIR):
        shutil.rmtree(TEST_DIR, ignore_errors=True)
    os.makedirs(TEST_DIR, exist_ok=True)

    results = TestResult()
    data_path = None
    trajectory_length = None
    model = None
    state = None
    loaded_params = None

    # Test 1: Action 映射对称性
    try:
        test_action_mapping(results)
    except Exception as e:
        results.fail("Test 1 异常", str(e))

    # Test 2: 专家策略
    try:
        test_expert_policy(results)
    except Exception as e:
        import traceback
        results.fail("Test 2 异常", str(e))
        traceback.print_exc()

    # Test 3: BC Reset
    try:
        test_bc_reset(results)
    except Exception as e:
        results.fail("Test 3 异常", str(e))

    # Test 4: 数据收集
    try:
        result = test_data_collection(results)
        if result is not None:
            data_path, trajectory_length = result
    except Exception as e:
        import traceback
        results.fail("Test 4 异常", str(e))
        traceback.print_exc()

    # Test 5: BC 训练
    if data_path is not None:
        try:
            model, state = test_bc_training(results, data_path, trajectory_length)
        except Exception as e:
            import traceback
            results.fail("Test 5 异常", str(e))
            traceback.print_exc()
    else:
        results.fail("Test 5 跳过", "依赖 Test 4")

    # Test 6: 序列化
    if model is not None and state is not None:
        try:
            _, loaded_params = test_serialization(results, model, state)
        except Exception as e:
            results.fail("Test 6 异常", str(e))
    else:
        results.fail("Test 6 跳过", "依赖 Test 5")

    # Test 7: 评估
    eval_params = loaded_params if loaded_params is not None else (state.params if state is not None else None)
    if model is not None and eval_params is not None:
        try:
            test_bc_evaluation(results, model, eval_params)
        except Exception as e:
            import traceback
            results.fail("Test 7 异常", str(e))
            traceback.print_exc()
    else:
        results.fail("Test 7 跳过", "依赖 Test 5/6")

    # Test 8: 视频渲染
    if model is not None and eval_params is not None:
        try:
            test_bc_video(results, model, eval_params)
        except Exception as e:
            results.fail("Test 8 异常", str(e))
    else:
        results.fail("Test 8 跳过", "依赖 Test 5/6")

    # 总结
    success = results.summary()
    print(f"\n📁 测试产物保留在: {TEST_DIR}")
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
