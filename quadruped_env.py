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
        max_push_force=150.0,   # Max push force in Newtons
        push_duration_steps=6,  # Duration of each push (steps)
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

        # Body IDs
        self.trunk_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "trunk")
        
        # Actuator and joint setup (12 DoF)
        self.num_joints = 12
        self.action_scale = 0.5  # scaling for target joint angle displacement
        
        # Nominal standing joint angles for Unitree A1
        # [FR_hip, FR_thigh, FR_calf, FL_hip, FL_thigh, FL_calf, RR_hip, RR_thigh, RR_calf, RL_hip, RL_thigh, RL_calf]
        self.default_qpos = np.array([
            0.0, 0.9, -1.8,
            0.0, 0.9, -1.8,
            0.0, 0.9, -1.8,
            0.0, 0.9, -1.8
        ], dtype=np.float32)

        # PD Controller gains
        self.kp = 35.0
        self.kd = 0.8
        self.max_torque = 33.5  # Max torque for Unitree A1 actuators (Nm)

        # Spaces:
        # Action: 12 target joint angles (residuals from default_qpos)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.num_joints,), dtype=np.float32
        )

        # Observation:
        # 3 (base ang vel) + 3 (projected gravity) + 3 (command: vx, vy, wz) + 12 (q - q0) + 12 (dq) + 12 (last_action) = 45
        obs_dim = 3 + 3 + 3 + 12 + 12 + 12
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # State buffers
        self.last_action = np.zeros(self.num_joints, dtype=np.float32)
        self.command = np.array([1.0, 0.0, 0.0], dtype=np.float32)  # [vx_cmd, vy_cmd, wz_cmd]
        self.step_count = 0
        self.max_episode_steps = 1000

        # Rendering
        self.render_mode = render_mode
        self.camera = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        # Initial pose (standing with slight randomization)
        self.data.qpos[2] = 0.28  # nominal trunk height (m)
        self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]  # orientation quaternion
        self.data.qpos[7:19] = self.default_qpos + np.random.uniform(-0.05, 0.05, size=12)
        self.data.qvel[:] = 0.0

        # Domain Randomization: Randomize trunk mass (+/- 1.5 kg)
        nominal_mass = self.model.body_mass[self.trunk_body_id]
        self.model.body_mass[self.trunk_body_id] = nominal_mass + np.random.uniform(-1.0, 2.0)

        # Domain Randomization: Ground friction
        self.model.geom_friction[:, 0] = np.random.uniform(0.4, 1.2)

        mujoco.mj_forward(self.model, self.data)

        # Reset states
        self.last_action[:] = 0.0
        self.push_counter = 0
        self.current_push_force[:] = 0.0
        self.step_count = 0

        # Randomized target velocity command: vx in [0.5, 1.5], vy in [-0.3, 0.3], wz in [-0.5, 0.5]
        self.command = np.array([
            np.random.uniform(0.6, 1.2),
            np.random.uniform(-0.2, 0.2),
            np.random.uniform(-0.4, 0.4)
        ], dtype=np.float32)

        return self._get_obs(), {}

    def _apply_perturbation(self):
        """Inject sudden external force impulses to the robot torso."""
        if self.push_counter > 0:
            # Continue active push
            self.data.xfrc_applied[self.trunk_body_id, :3] = self.current_push_force
            self.push_counter -= 1
        else:
            # Reset applied force
            self.data.xfrc_applied[self.trunk_body_id, :] = 0.0
            # Check if a new perturbation should trigger
            if np.random.rand() < self.perturbation_prob:
                angle = np.random.uniform(0, 2 * np.pi)
                magnitude = np.random.uniform(50.0, self.max_push_force)
                self.current_push_force[0] = magnitude * np.cos(angle)
                self.current_push_force[1] = magnitude * np.sin(angle)
                self.current_push_force[2] = np.random.uniform(-20.0, 20.0)
                self.push_counter = self.push_duration_steps

    def step(self, action):
        self.step_count += 1
        action = np.clip(action, -1.0, 1.0)
        target_qpos = self.default_qpos + action * self.action_scale

        # MuJoCo Menagerie Unitree A1 uses built-in position actuators (<position kp="100"/>)
        # target_qpos directly represents desired joint angles (in radians)
        self.data.ctrl[:12] = target_qpos

        applied_torques = []
        for _ in range(self.sim_substeps):
            self._apply_perturbation()
            mujoco.mj_step(self.model, self.data)
            applied_torques.append(self.data.actuator_force[:12].copy())

        # Compute observation, reward, and termination
        obs = self._get_obs()
        reward, info = self._compute_reward(action, np.mean(applied_torques, axis=0))
        
        # Check termination (robot tipped over or fell)
        trunk_height = self.data.qpos[2]
        projected_gravity = obs[3:6]
        
        terminated = False
        if trunk_height < 0.16 or trunk_height > 0.45:
            terminated = True
        if projected_gravity[2] > -0.5:  # tilted more than ~60 deg
            terminated = True

        truncated = self.step_count >= self.max_episode_steps
        self.last_action = action.copy()

        info["is_push_active"] = (self.push_counter > 0)
        info["trunk_height"] = trunk_height

        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        # Base angular velocity in robot local frame
        base_angvel = self.data.sensor("angular-velocity").data if "angular-velocity" in [self.model.sensor(i).name for i in range(self.model.nsensor)] else self.data.qvel[3:6]

        # Projected gravity vector in robot frame: R_world_to_robot * [0, 0, -1]
        rot_mat = np.zeros(9)
        mujoco.mju_quat2Mat(rot_mat, self.data.qpos[3:7])
        rot_mat = rot_mat.reshape((3, 3))
        proj_gravity = rot_mat.T @ np.array([0.0, 0.0, -1.0])

        # Joint position error relative to nominal pose and velocities
        joint_pos_err = (self.data.qpos[7:19] - self.default_qpos)
        joint_vel = self.data.qvel[6:18]

        obs = np.concatenate([
            base_angvel,
            proj_gravity,
            self.command,
            joint_pos_err,
            joint_vel,
            self.last_action
        ], dtype=np.float32)

        return np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)

    def _compute_reward(self, action, torques):
        # Base linear velocity in world frame
        base_linvel = self.data.qvel[0:3]
        base_angvel = self.data.qvel[3:6]
        
        # Projected gravity
        rot_mat = np.zeros(9)
        mujoco.mju_quat2Mat(rot_mat, self.data.qpos[3:7])
        rot_mat = rot_mat.reshape((3, 3))
        local_linvel = rot_mat.T @ base_linvel

        # 1. Forward velocity tracking reward
        vel_err = (local_linvel[0] - self.command[0]) ** 2
        r_vel = np.exp(-vel_err / 0.25)

        # 2. Lateral velocity penalty
        r_lateral = -1.0 * (local_linvel[1] - self.command[1]) ** 2

        # 3. Angular yaw rate tracking reward
        yaw_err = (base_angvel[2] - self.command[2]) ** 2
        r_yaw = np.exp(-yaw_err / 0.25)

        # 4. Base posture penalty (maintain horizontal torso)
        proj_gravity = rot_mat.T @ np.array([0.0, 0.0, -1.0])
        r_orient = -2.0 * np.sum(np.square(proj_gravity[:2]))

        # 5. Height maintenance reward
        r_height = -5.0 * (self.data.qpos[2] - 0.28) ** 2

        # 6. Action smoothness & energy penalties
        r_smooth = -0.05 * np.sum(np.square(action - self.last_action))
        r_torque = -0.0001 * np.sum(np.square(torques))

        # 7. Survival / Healthy bonus
        r_alive = 0.5

        total_reward = (
            1.5 * r_vel +
            0.5 * r_yaw +
            r_lateral +
            r_orient +
            r_height +
            r_smooth +
            r_torque +
            r_alive
        )

        reward_info = {
            "r_vel": r_vel,
            "r_yaw": r_yaw,
            "r_orient": r_orient,
            "r_height": r_height,
        }

        return float(total_reward), reward_info
