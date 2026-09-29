"""
Gymnasium MuJoCo Environment for Quadruped Gait Adaptation under External Perturbations.
Compatible with Unitree A1/Go1 robots from MuJoCo Menagerie.
"""
import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco

class QuadrupedPerturbationEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array", "depth_array"], "render_fps": 50}

    def __init__(
        self,
        model_path=None,
        control_dt=0.02,        # 50 Hz control loop
        sim_dt=0.002,           # 500 Hz physics simulation
        perturbation_prob=0.02, # Probability of triggering a push perturbation per step
        max_push_force=60.0,    # Max push force in Newtons (realistic for 12kg A1)
        push_duration_steps=6,  # Duration of each push (steps)
        gait_freq=2.2,          # Trotting gait frequency in Hz
        render_mode=None,
    ):
        super().__init__()
        
        if model_path is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(base_dir, "models", "unitree_a1", "scene.xml")
            
        if not os.path.exists(model_path):
            from setup_robot import setup_robot_model
            model_path = setup_robot_model()

        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        # Simulation timings
        self.sim_dt = sim_dt
        self.control_dt = control_dt
        self.model.opt.timestep = self.sim_dt
        self.sim_substeps = int(round(self.control_dt / self.sim_dt))

        # Perturbation configuration
        self.perturbation_prob = perturbation_prob
        self.max_push_force = max_push_force
        self.push_duration_steps = push_duration_steps
        self.push_counter = 0
        self.current_push_force = np.zeros(3)

        # Body IDs and nominal mass
        self.trunk_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "trunk")
        self.nominal_trunk_mass = float(self.model.body_mass[self.trunk_body_id])
        
        # Actuator and joint setup (12 DoF)
        self.num_joints = 12
        # Per-joint residual action scaling: [hip_abduction, thigh, calf] x 4 legs
        self.action_scale = np.array([
            0.15, 0.25, 0.25,
            0.15, 0.25, 0.25,
            0.15, 0.25, 0.25,
            0.15, 0.25, 0.25
        ], dtype=np.float32)
        
        # Nominal standing joint angles for Unitree A1
        # [FR_hip, FR_thigh, FR_calf, FL_hip, FL_thigh, FL_calf, RR_hip, RR_thigh, RR_calf, RL_hip, RL_thigh, RL_calf]
        self.default_qpos = np.array([
            0.0, 0.9, -1.8,
            0.0, 0.9, -1.8,
            0.0, 0.9, -1.8,
            0.0, 0.9, -1.8
        ], dtype=np.float32)

        # Trotting gait parameters (FR+RL in phase 0, FL+RR in phase pi)
        self.gait_freq = gait_freq
        self.leg_phase_offsets = np.array([0.0, np.pi, np.pi, 0.0], dtype=np.float32)
        self.gait_phase = 0.0

        # Spaces:
        # Action: 12 residual joint angle adjustments on top of the cyclic trotting reference
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.num_joints,), dtype=np.float32
        )

        # Observation (47 dims):
        # 3 (base ang vel) + 3 (projected gravity) + 3 (command: vx, vy, wz)
        # + 2 (gait clock: sin, cos) + 12 (q - q_ref) + 12 (dq) + 12 (last_action) = 47
        obs_dim = 3 + 3 + 3 + 2 + 12 + 12 + 12
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # State buffers
        self.last_action = np.zeros(self.num_joints, dtype=np.float32)
        self.smoothed_action = np.zeros(self.num_joints, dtype=np.float32)
        self.command = np.array([0.8, 0.0, 0.0], dtype=np.float32)  # [vx_cmd, vy_cmd, wz_cmd]
        self.step_count = 0
        self.max_episode_steps = 1000
        self.prev_x_pos = 0.0

        # Rendering
        self.render_mode = render_mode
        self.camera = None

    def _get_gait_reference(self):
        """
        Compute kinematic reference joint angles for a diagonal trotting gait.
        In Unitree A1:
        - Decreasing thigh angle (q_thigh < 0.9) swings the foot FORWARD (+X).
        - Increasing thigh angle (q_thigh > 0.9) sweeps the foot BACKWARD (-X), propelling body forward.
        - Increasing thigh (+lift) and bending calf (-2*lift) lifts the foot vertically (+Z) during swing.
        """
        q_ref = self.default_qpos.copy()
        vx_cmd = float(self.command[0])
        vy_cmd = float(self.command[1])
        wz_cmd = float(self.command[2])

        # Scale stride with commanded speed
        speed_scale = np.clip(abs(vx_cmd) / 0.8, 0.4, 1.4)
        direction = 1.0 if vx_cmd >= 0.0 else -1.0

        swing_amp = 0.26 * speed_scale * direction  # rad (~15 deg forward/back thigh sweep)
        lift_amp = 0.24                             # rad (~8 cm vertical foot clearance during swing)
        hip_amp = 0.08 * np.clip(vy_cmd / 0.3, -1.0, 1.0)

        for leg_idx in range(4):
            theta = (self.gait_phase + self.leg_phase_offsets[leg_idx]) % (2.0 * np.pi)
            # Swing phase is theta in [0, pi] where sin(theta) > 0
            swing_signal = max(0.0, np.sin(theta))
            stance_press = min(0.0, np.sin(theta))  # <= 0 during stance [pi, 2*pi]

            # Yaw differential stride adjustment (left legs vs right legs)
            is_left_leg = (leg_idx in [1, 3])
            yaw_mod = (0.06 * wz_cmd) if not is_left_leg else (-0.06 * wz_cmd)

            # At theta=0 (liftoff), cos(0)=+1 -> foot at rear (0.9 + swing_amp)
            # At theta=pi (touchdown), cos(pi)=-1 -> foot at front (0.9 - swing_amp)
            # During stance [pi, 2*pi], cos(theta) goes -1 -> +1 (sweeps front to back, propelling +X)
            thigh_offset = (swing_amp + yaw_mod) * np.cos(theta) + lift_amp * swing_signal
            calf_offset = -2.0 * lift_amp * swing_signal + 0.05 * stance_press
            hip_offset = hip_amp * np.cos(theta)

            base_idx = leg_idx * 3
            q_ref[base_idx + 0] += hip_offset
            q_ref[base_idx + 1] += thigh_offset
            q_ref[base_idx + 2] += calf_offset

        return q_ref

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        # Reset gait phase and command first
        self.gait_phase = np.random.uniform(0.0, 2.0 * np.pi)
        self.command = np.array([
            np.random.uniform(0.65, 1.0),
            np.random.uniform(-0.1, 0.1),
            np.random.uniform(-0.2, 0.2)
        ], dtype=np.float32)

        # Initial pose aligned with the starting gait phase for smooth liftoff
        q_init = self._get_gait_reference()
        self.data.qpos[0:2] = 0.0
        self.data.qpos[2] = 0.28  # nominal trunk height (m)
        self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]  # orientation quaternion
        self.data.qpos[7:19] = q_init + np.random.uniform(-0.02, 0.02, size=12)
        self.data.qvel[:] = 0.0
        self.data.qvel[0] = 0.3  # slight initial forward momentum

        # Domain Randomization: Randomize trunk mass around nominal_trunk_mass (avoid cumulative drift!)
        self.model.body_mass[self.trunk_body_id] = max(
            2.0, self.nominal_trunk_mass + np.random.uniform(-0.8, 1.2)
        )

        # Domain Randomization: Ground friction (high enough for traction without slipping in place)
        self.model.geom_friction[:, 0] = np.random.uniform(0.7, 1.3)

        mujoco.mj_forward(self.model, self.data)

        # Reset states
        self.last_action[:] = 0.0
        self.smoothed_action[:] = 0.0
        self.push_counter = 0
        self.current_push_force[:] = 0.0
        self.step_count = 0
        self.prev_x_pos = float(self.data.qpos[0])

        return self._get_obs(), {}

    def set_max_push_force(self, force: float):
        """Update maximum push force dynamically for curriculum training."""
        self.max_push_force = float(force)

    def _apply_perturbation(self):
        """Inject sudden external force impulses to the robot torso."""
        if self.push_counter > 0:
            # Continue active push
            self.data.xfrc_applied[self.trunk_body_id, :3] = self.current_push_force
            self.push_counter -= 1
        else:
            # Reset applied force
            self.data.xfrc_applied[self.trunk_body_id, :] = 0.0
            # Trigger new perturbation only after initial 25 steps of walking
            if self.step_count > 25 and self.max_push_force > 1.0 and np.random.rand() < self.perturbation_prob:
                angle = np.random.uniform(0, 2 * np.pi)
                min_f = min(15.0, self.max_push_force * 0.4)
                magnitude = np.random.uniform(min_f, self.max_push_force)
                self.current_push_force[0] = magnitude * np.cos(angle)
                self.current_push_force[1] = magnitude * np.sin(angle)
                self.current_push_force[2] = np.random.uniform(-5.0, 5.0)
                self.push_counter = self.push_duration_steps

    def step(self, action):
        self.step_count += 1
        action = np.clip(action, -1.0, 1.0)

        # Low-pass filter residual actions to eliminate high-frequency jitter
        self.smoothed_action = 0.65 * action + 0.35 * self.smoothed_action

        # Advance gait clock
        self.gait_phase = (self.gait_phase + 2.0 * np.pi * self.gait_freq * self.control_dt) % (2.0 * np.pi)
        q_ref = self._get_gait_reference()

        # Target joint positions = cyclic trotting gait + RL residual adjustments
        target_qpos = q_ref + self.smoothed_action * self.action_scale

        # MuJoCo Menagerie Unitree A1 uses built-in position actuators (<position kp="100"/>)
        self.data.ctrl[:12] = target_qpos

        # Apply perturbation force at the control step level
        self._apply_perturbation()

        applied_torques = []
        for _ in range(self.sim_substeps):
            mujoco.mj_step(self.model, self.data)
            applied_torques.append(self.data.actuator_force[:12].copy())

        # Compute observation, reward, and termination
        obs = self._get_obs(q_ref)
        reward, info = self._compute_reward(action, q_ref, np.mean(applied_torques, axis=0))
        
        # Check termination (robot tipped over or collapsed)
        trunk_height = float(self.data.qpos[2])
        projected_gravity = obs[3:6]
        
        terminated = False
        if trunk_height < 0.16 or trunk_height > 0.45:
            terminated = True
        if projected_gravity[2] > -0.45:  # tilted more than ~63 deg
            terminated = True

        if terminated:
            reward -= 5.0  # Clear penalty for falling so policy prioritizes balance recovery

        truncated = self.step_count >= self.max_episode_steps
        self.last_action = action.copy()
        self.prev_x_pos = float(self.data.qpos[0])

        info["is_push_active"] = (self.push_counter > 0)
        info["trunk_height"] = trunk_height
        info["x_position"] = float(self.data.qpos[0])

        return obs, float(reward), terminated, truncated, info

    def _get_obs(self, q_ref=None):
        if q_ref is None:
            q_ref = self._get_gait_reference()

        # Base angular velocity in robot local frame
        base_angvel = self.data.sensor("angular-velocity").data if "angular-velocity" in [self.model.sensor(i).name for i in range(self.model.nsensor)] else self.data.qvel[3:6]

        # Projected gravity vector in robot frame: R_world_to_robot * [0, 0, -1]
        rot_mat = np.zeros(9)
        mujoco.mju_quat2Mat(rot_mat, self.data.qpos[3:7])
        rot_mat = rot_mat.reshape((3, 3))
        proj_gravity = rot_mat.T @ np.array([0.0, 0.0, -1.0])

        # Gait phase clock
        clock = np.array([np.sin(self.gait_phase), np.cos(self.gait_phase)], dtype=np.float32)

        # Joint position error relative to current gait phase reference
        joint_pos_err = (self.data.qpos[7:19] - q_ref)
        joint_vel = self.data.qvel[6:18] * 0.1  # scaled for neural network stability

        obs = np.concatenate([
            base_angvel,
            proj_gravity,
            self.command,
            clock,
            joint_pos_err,
            joint_vel,
            self.last_action
        ], dtype=np.float32)

        return np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)

    def _compute_reward(self, action, q_ref, torques):
        # Base linear and angular velocity
        base_linvel = self.data.qvel[0:3]
        base_angvel = self.data.qvel[3:6]
        
        # Local frame transformation
        rot_mat = np.zeros(9)
        mujoco.mju_quat2Mat(rot_mat, self.data.qpos[3:7])
        rot_mat = rot_mat.reshape((3, 3))
        local_linvel = rot_mat.T @ base_linvel
        proj_gravity = rot_mat.T @ np.array([0.0, 0.0, -1.0])

        # 1. Forward velocity tracking reward
        vel_err = (local_linvel[0] - self.command[0]) ** 2
        r_vel = np.exp(-vel_err / 0.20)

        # 2. Cell-by-cell forward displacement & progress reward
        dx = float(self.data.qpos[0]) - self.prev_x_pos
        r_forward = 2.5 * np.clip(local_linvel[0], -0.3, self.command[0] * 1.2) + 15.0 * dx
        # Anti-stagnation penalty: strongly discourage standing in place
        if self.command[0] > 0.2 and local_linvel[0] < 0.15:
            r_forward -= 1.0

        # 3. Trotting gait trajectory tracking reward (keeps crisp alternating steps)
        gait_err = np.sum(np.square(self.data.qpos[7:19] - q_ref))
        r_gait = np.exp(-gait_err / 0.45)

        # 4. Lateral drift & vertical bounce penalties
        r_lateral = -1.5 * ((local_linvel[1] - self.command[1]) ** 2) - 0.8 * (base_linvel[2] ** 2)

        # 5. Angular yaw tracking and roll/pitch rate damping
        yaw_err = (base_angvel[2] - self.command[2]) ** 2
        r_yaw = np.exp(-yaw_err / 0.25)
        r_angvel_xy = -0.05 * np.sum(np.square(base_angvel[:2]))

        # 6. Base posture penalty (maintain horizontal torso roll/pitch)
        r_orient = -2.5 * np.sum(np.square(proj_gravity[:2]))

        # 7. Height maintenance around nominal 0.28m
        height_err = max(0.0, abs(self.data.qpos[2] - 0.28) - 0.025)
        r_height = -8.0 * (height_err ** 2)

        # 8. Residual regularization & smoothness (use small residuals unless recovering from push)
        r_smooth = -0.01 * np.sum(np.square(action - self.last_action)) - 0.01 * np.sum(np.square(action))
        r_torque = -0.00003 * np.sum(np.square(torques))

        # 9. Survival bonus
        r_alive = 0.2

        total_reward = (
            3.0 * r_vel +
            r_forward +
            2.0 * r_gait +
            0.6 * r_yaw +
            r_lateral +
            r_angvel_xy +
            r_orient +
            r_height +
            r_smooth +
            r_torque +
            r_alive
        )

        reward_info = {
            "r_vel": r_vel,
            "r_forward": r_forward,
            "r_gait": r_gait,
            "r_yaw": r_yaw,
            "r_orient": r_orient,
            "r_height": r_height,
        }

        return float(total_reward), reward_info


