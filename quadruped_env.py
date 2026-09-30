"""
Gymnasium MuJoCo Environment for Quadruped Gait Adaptation under External Perturbations
with 360-Degree Sharp-Turn Random Cell Bait Navigation.
Compatible with Unitree A1/Go1 robots from MuJoCo Menagerie.
"""
import os
import re
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco


def _ensure_bait_scene_xml(scene_xml_path: str) -> str:
    """
    Create a scene_bait.xml alongside scene.xml that injects a visual-only mocap body
    for the target cell pad and glowing 3D bait sphere.
    Using a mocap body with contype=0/conaffinity=0 preserves exact qpos/qvel dimensions.
    """
    scene_dir = os.path.dirname(os.path.abspath(scene_xml_path))
    bait_xml_path = os.path.join(scene_dir, "scene_bait.xml")

    with open(scene_xml_path, "r", encoding="utf-8") as f:
        xml_content = f.read()

    if 'name="bait"' in xml_content:
        return scene_xml_path

    bait_body_xml = """
    <!-- Visual 3D Bait and Target Grid Cell Marker (Mocap, No Physics Collision) -->
    <body name="bait" mocap="true" pos="2.0 0.0 0.0">
      <geom name="bait_cell_pad" type="box" size="0.40 0.40 0.004" pos="0 0 0.004"
            rgba="1.0 0.85 0.10 0.40" contype="0" conaffinity="0"/>
      <geom name="bait_ring" type="cylinder" size="0.26 0.008" pos="0 0 0.010"
            rgba="1.0 0.35 0.05 0.80" contype="0" conaffinity="0"/>
      <geom name="bait_sphere" type="sphere" size="0.12" pos="0 0 0.20"
            rgba="1.0 0.12 0.12 0.95" contype="0" conaffinity="0"/>
    </body>
  </worldbody>"""

    modified_xml = re.sub(r"</worldbody>", bait_body_xml, xml_content, count=1)
    with open(bait_xml_path, "w", encoding="utf-8") as f:
        f.write(modified_xml)

    return bait_xml_path


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
        use_bait=True,          # Enable 360-degree random grid cell bait navigation
        cell_size=1.0,          # Grid cell size in meters (1.0m x 1.0m cells)
        bait_reach_radius=0.42, # Distance threshold (m) to collect the bait in the target cell
        render_mode=None,
    ):
        super().__init__()

        if model_path is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(base_dir, "models", "unitree_a1", "scene.xml")

        if not os.path.exists(model_path):
            from setup_robot import setup_robot_model
            model_path = setup_robot_model()

        model_path = _ensure_bait_scene_xml(model_path)

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

        # Bait & Grid Cell configuration
        self.use_bait = use_bait
        self.cell_size = float(cell_size)
        self.bait_reach_radius = float(bait_reach_radius)
        self.bait_pos = np.array([2.0, 0.0], dtype=np.float32)
        self.prev_bait_dist = 2.0
        self.baits_collected = 0

        # Locate bait mocap ID if present
        bait_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "bait")
        if bait_body_id >= 0 and self.model.body_mocapid[bait_body_id] >= 0:
            self.bait_mocap_id = int(self.model.body_mocapid[bait_body_id])
        else:
            self.bait_mocap_id = -1

        # Body IDs and nominal mass
        self.trunk_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "trunk")
        self.nominal_trunk_mass = float(self.model.body_mass[self.trunk_body_id])

        # Actuator and joint setup (12 DoF)
        self.num_joints = 12
        # Per-joint residual action scaling: [hip_abduction, thigh, calf] x 4 legs
        self.action_scale = np.array([
            0.18, 0.26, 0.26,
            0.18, 0.26, 0.26,
            0.18, 0.26, 0.26,
            0.18, 0.26, 0.26
        ], dtype=np.float32)

        # Nominal standing joint angles for Unitree A1
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

        # Observation (47 dims - 100% compatible with existing checkpoints):
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

    def _get_robot_yaw(self) -> float:
        """Compute robot trunk yaw angle (rad) in world frame from quaternion."""
        w, x, y, z = self.data.qpos[3:7]
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return float(np.arctan2(siny_cosp, cosy_cosp))

    def _spawn_random_bait(self):
        """
        Place the bait at the center of a random floor grid cell (360 degrees around the robot).
        """
        robot_x, robot_y = float(self.data.qpos[0]), float(self.data.qpos[1])

        # Pick a random 360-degree direction and radial distance (1.5m to 3.5m)
        angle = np.random.uniform(-np.pi, np.pi)
        radius = np.random.uniform(1.5, 3.2)

        raw_x = robot_x + radius * np.cos(angle)
        raw_y = robot_y + radius * np.sin(angle)

        # Snap to the center of a discrete floor grid cell (e.g., 1.0m x 1.0m grid)
        cell_x = np.round(raw_x / self.cell_size) * self.cell_size
        cell_y = np.round(raw_y / self.cell_size) * self.cell_size

        # Ensure new cell is at least 1.2m away from current robot position
        if np.hypot(cell_x - robot_x, cell_y - robot_y) < 1.2:
            cell_x += self.cell_size * (1.0 if np.cos(angle) >= 0 else -1.0)
            cell_y += self.cell_size * (1.0 if np.sin(angle) >= 0 else -1.0)

        self.bait_pos = np.array([cell_x, cell_y], dtype=np.float32)
        self.prev_bait_dist = float(np.hypot(self.bait_pos[0] - robot_x, self.bait_pos[1] - robot_y))

        # Update visual mocap body position in MuJoCo scene
        if self.bait_mocap_id >= 0:
            self.data.mocap_pos[self.bait_mocap_id] = [self.bait_pos[0], self.bait_pos[1], 0.0]

    def _update_bait_command(self):
        """
        Compute 360-degree closed-loop steering command [vx_cmd, vy_cmd, wz_cmd]
        to turn toward and walk into the target bait cell.
        """
        if not self.use_bait:
            return

        robot_x, robot_y = float(self.data.qpos[0]), float(self.data.qpos[1])
        dx = float(self.bait_pos[0] - robot_x)
        dy = float(self.bait_pos[1] - robot_y)
        dist = float(np.hypot(dx, dy))

        target_yaw = float(np.arctan2(dy, dx))
        robot_yaw = self._get_robot_yaw()

        # Wrap heading error to [-pi, pi]
        yaw_err = (target_yaw - robot_yaw + np.pi) % (2.0 * np.pi) - np.pi

        # 360° Sharp-Turn Controller:
        # 1. High-authority yaw rate command up to +/- 1.2 rad/s
        wz_cmd = float(np.clip(1.8 * yaw_err, -1.2, 1.2))

        # 2. Modulate forward speed based on heading alignment:
        #    When bait is behind/sideways (|yaw_err| > 45 deg), slow forward speed to pivot sharply in place.
        #    When aligned with bait, accelerate to full 0.85 m/s trot.
        alignment = max(0.20, float(np.cos(yaw_err)))
        desired_speed = float(np.clip(0.85 * min(1.0, dist / 0.5), 0.30, 0.90))
        vx_cmd = desired_speed * alignment

        # 3. Slight lateral stepping assist toward the bait
        vy_cmd = float(np.clip(0.25 * np.sin(yaw_err), -0.20, 0.20))

        self.command = np.array([vx_cmd, vy_cmd, wz_cmd], dtype=np.float32)

    def _get_gait_reference(self):
        """
        Compute kinematic reference joint angles for a diagonal trotting gait with
        360-degree sharp-turning differential stride and hip abduction modulation.
        """
        q_ref = self.default_qpos.copy()
        vx_cmd = float(self.command[0])
        vy_cmd = float(self.command[1])
        wz_cmd = float(self.command[2])

        # Keep legs actively stepping both when walking forward AND when pivoting in place
        speed_scale = np.clip(abs(vx_cmd) / 0.8, 0.25, 1.4)
        turn_activity = np.clip(abs(wz_cmd) / 1.0, 0.0, 1.0)
        step_activity = max(speed_scale, 0.75 * turn_activity)
        direction = 1.0 if vx_cmd >= 0.0 else -1.0

        swing_amp = 0.26 * speed_scale * direction
        lift_amp = 0.24 * np.clip(step_activity + 0.3, 0.7, 1.15)
        hip_lat_amp = 0.09 * np.clip(vy_cmd / 0.25, -1.0, 1.0)

        for leg_idx in range(4):
            theta = (self.gait_phase + self.leg_phase_offsets[leg_idx]) % (2.0 * np.pi)
            swing_signal = max(0.0, np.sin(theta))
            stance_press = min(0.0, np.sin(theta))

            # Differential stride between right legs (0, 2) and left legs (1, 3) for sharp yaw turning
            is_left_leg = (leg_idx in [1, 3])
            is_front_leg = (leg_idx in [0, 1])
            yaw_stride_mod = (0.15 * wz_cmd) if not is_left_leg else (-0.15 * wz_cmd)

            # Front vs Rear hip abduction torque for rapid 360-degree pivoting
            hip_turn_mod = (0.07 * wz_cmd) if is_front_leg else (-0.07 * wz_cmd)

            thigh_offset = (swing_amp + yaw_stride_mod) * np.cos(theta) + lift_amp * swing_signal
            calf_offset = -2.0 * lift_amp * swing_signal + 0.05 * stance_press
            hip_offset = (hip_lat_amp + hip_turn_mod) * np.cos(theta)

            base_idx = leg_idx * 3
            q_ref[base_idx + 0] += hip_offset
            q_ref[base_idx + 1] += thigh_offset
            q_ref[base_idx + 2] += calf_offset

        return q_ref

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        # Reset gait phase
        self.gait_phase = np.random.uniform(0.0, 2.0 * np.pi)
        self.baits_collected = 0

        # Initial robot pose at origin
        self.data.qpos[0:2] = 0.0
        self.data.qpos[2] = 0.28  # nominal trunk height (m)
        self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]

        if self.use_bait:
            self._spawn_random_bait()
            self._update_bait_command()
        else:
            self.command = np.array([
                np.random.uniform(0.65, 1.0),
                np.random.uniform(-0.15, 0.15),
                np.random.uniform(-0.8, 0.8)
            ], dtype=np.float32)
            if self.bait_mocap_id >= 0:
                self.data.mocap_pos[self.bait_mocap_id] = [100.0, 100.0, -5.0]

        q_init = self._get_gait_reference()
        self.data.qpos[7:19] = q_init + np.random.uniform(-0.02, 0.02, size=12)
        self.data.qvel[:] = 0.0
        self.data.qvel[0] = 0.25

        # Domain Randomization: Trunk mass around nominal_trunk_mass
        self.model.body_mass[self.trunk_body_id] = max(
            2.0, self.nominal_trunk_mass + np.random.uniform(-0.8, 1.2)
        )

        # Domain Randomization: Ground friction
        self.model.geom_friction[:, 0] = np.random.uniform(0.7, 1.3)

        mujoco.mj_forward(self.model, self.data)

        # Reset state buffers
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
            self.data.xfrc_applied[self.trunk_body_id, :3] = self.current_push_force
            self.push_counter -= 1
        else:
            self.data.xfrc_applied[self.trunk_body_id, :] = 0.0
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

        # Update closed-loop steering command toward current bait cell
        if self.use_bait:
            self._update_bait_command()

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
            reward -= 5.0

        truncated = self.step_count >= self.max_episode_steps
        self.last_action = action.copy()
        self.prev_x_pos = float(self.data.qpos[0])

        info["is_push_active"] = (self.push_counter > 0)
        info["trunk_height"] = trunk_height
        info["x_position"] = float(self.data.qpos[0])
        info["y_position"] = float(self.data.qpos[1])
        info["bait_pos"] = self.bait_pos.copy()
        info["baits_collected"] = self.baits_collected

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
        joint_vel = self.data.qvel[6:18] * 0.1

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
        base_linvel = self.data.qvel[0:3]
        base_angvel = self.data.qvel[3:6]

        rot_mat = np.zeros(9)
        mujoco.mju_quat2Mat(rot_mat, self.data.qpos[3:7])
        rot_mat = rot_mat.reshape((3, 3))
        local_linvel = rot_mat.T @ base_linvel
        local_angvel = rot_mat.T @ base_angvel
        proj_gravity = rot_mat.T @ np.array([0.0, 0.0, -1.0])

        # 1. Forward velocity tracking reward (in robot local frame)
        vel_err = (local_linvel[0] - self.command[0]) ** 2
        r_vel = np.exp(-vel_err / 0.20)

        # 2. Progress toward target (either bait cell in 2D or +X track)
        bait_reached = False
        if self.use_bait:
            robot_x, robot_y = float(self.data.qpos[0]), float(self.data.qpos[1])
            curr_bait_dist = float(np.hypot(self.bait_pos[0] - robot_x, self.bait_pos[1] - robot_y))
            d_bait = self.prev_bait_dist - curr_bait_dist
            self.prev_bait_dist = curr_bait_dist

            r_forward = 2.5 * np.clip(local_linvel[0], -0.2, self.command[0] * 1.2) + 18.0 * d_bait

            # Check if robot entered the bait cell!
            if curr_bait_dist <= self.bait_reach_radius:
                bait_reached = True
                self.baits_collected += 1
                r_forward += 25.0  # Bonus for collecting the bait cell
                self._spawn_random_bait()
        else:
            dx = float(self.data.qpos[0]) - self.prev_x_pos
            r_forward = 2.5 * np.clip(local_linvel[0], -0.3, self.command[0] * 1.2) + 15.0 * dx
            if self.command[0] > 0.2 and local_linvel[0] < 0.15:
                r_forward -= 1.0

        # 3. Trotting gait trajectory tracking reward
        gait_err = np.sum(np.square(self.data.qpos[7:19] - q_ref))
        r_gait = np.exp(-gait_err / 0.45)

        # 4. Lateral velocity tracking & vertical bounce penalties
        r_lateral = -1.5 * ((local_linvel[1] - self.command[1]) ** 2) - 0.8 * (base_linvel[2] ** 2)

        # 5. Sharp-turn angular yaw rate tracking (strong weight for 360-degree turning)
        yaw_err = (local_angvel[2] - self.command[2]) ** 2
        r_yaw = np.exp(-yaw_err / 0.30)
        r_angvel_xy = -0.05 * np.sum(np.square(local_angvel[:2]))

        # 6. Base posture penalty (maintain horizontal torso roll/pitch)
        r_orient = -2.5 * np.sum(np.square(proj_gravity[:2]))

        # 7. Height maintenance around nominal 0.28m
        height_err = max(0.0, abs(self.data.qpos[2] - 0.28) - 0.025)
        r_height = -8.0 * (height_err ** 2)

        # 8. Residual regularization & smoothness
        r_smooth = -0.01 * np.sum(np.square(action - self.last_action)) - 0.01 * np.sum(np.square(action))
        r_torque = -0.00003 * np.sum(np.square(torques))

        # 9. Survival bonus
        r_alive = 0.2

        total_reward = (
            3.0 * r_vel +
            r_forward +
            2.0 * r_gait +
            1.5 * r_yaw +
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
            "bait_reached": bait_reached,
        }

        return float(total_reward), reward_info
