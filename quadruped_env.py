"""
Gymnasium MuJoCo Environment for Quadruped Gait Adaptation under External Perturbations.
Features:
  1. 360-Degree Random Cell Bait Navigation (Goal-Directed Locomotion)
  2. 3-in-1 Multi-Gait Controller (0 = 4-Beat Walk, 1 = Diagonal Trot, 2 = High-Speed Bound)
  3. 3-Legged Fault-Tolerant Limp-Mode Adaptation (Actuator / Leg Failure Recovery)
  4. Proprioceptive History / RMA (Rapid Motor Adaptation) 10-Step Sliding Window
Compatible with Unitree A1/Go1 robots from MuJoCo Menagerie.
"""
import os
import re
from collections import deque
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco


def _ensure_bait_scene_xml(scene_xml_path: str) -> str:
    """
    Create a scene_bait.xml alongside scene.xml that injects visual-only mocap bodies
    for the target grid cell bait and an external push/fault indicator.
    Uses atomic temp-file replacement so parallel SubprocVecEnv workers never read a truncated file.
    """
    scene_dir = os.path.dirname(os.path.abspath(scene_xml_path))
    bait_xml_path = os.path.join(scene_dir, "scene_bait.xml")

    # Fast path: if scene_bait.xml already exists and is complete, return immediately without rewriting
    if os.path.exists(bait_xml_path) and os.path.getsize(bait_xml_path) > 100:
        try:
            with open(bait_xml_path, "r", encoding="utf-8") as bf:
                existing = bf.read()
            if 'name="bait"' in existing and 'name="fault_marker"' in existing and "</mujoco>" in existing:
                return bait_xml_path
        except OSError:
            pass

    with open(scene_xml_path, "r", encoding="utf-8") as f:
        xml_content = f.read()

    if 'name="bait"' in xml_content and 'name="fault_marker"' in xml_content:
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
    <!-- Visual Indicator above Broken Leg when 3-Legged Fault Mode is Active -->
    <body name="fault_marker" mocap="true" pos="100.0 100.0 -5.0">
      <geom name="fault_beacon" type="sphere" size="0.065" pos="0 0 0.12"
            rgba="1.0 0.0 0.8 0.90" contype="0" conaffinity="0"/>
    </body>
  </worldbody>"""

    modified_xml = re.sub(r"</worldbody>", bait_body_xml, xml_content, count=1)

    # Write to a PID-unique temp file and atomically replace scene_bait.xml
    tmp_path = f"{bait_xml_path}.tmp.{os.getpid()}"
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(modified_xml)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, bait_xml_path)

    return bait_xml_path


class QuadrupedPerturbationEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array", "depth_array"], "render_fps": 50}

    # Gait Mode Constants
    GAIT_WALK = 0   # 4-Beat Crawl/Walk: [0, pi, pi/2, 3*pi/2] (High stability / sharp turns)
    GAIT_TROT = 1   # 2-Beat Diagonal Trot: [0, pi, pi, 0] (Balanced medium-speed cruising)
    GAIT_BOUND = 2  # 2-Beat High-Speed Bound: [0, 0, pi, pi] (Sprint when aligned with distant bait)
    GAIT_NAMES = {0: "WALK", 1: "TROT", 2: "BOUND"}
    LEG_NAMES = {0: "FR", 1: "FL", 2: "RR", 3: "RL"}

    def __init__(
        self,
        model_path=None,
        control_dt=0.02,        # 50 Hz control loop
        sim_dt=0.002,           # 500 Hz physics simulation
        perturbation_prob=0.02, # Probability of triggering a push perturbation per step
        max_push_force=60.0,    # Max push force in Newtons (realistic for 12kg A1)
        push_duration_steps=6,  # Duration of each push (steps)
        gait_freq=2.2,          # Nominal trotting gait frequency in Hz
        use_bait=True,          # Enable 360-degree random grid cell bait navigation
        cell_size=1.0,          # Grid cell size in meters (1.0m x 1.0m cells)
        bait_reach_radius=0.42, # Distance threshold (m) to collect the bait in the target cell
        leg_failure_prob=0.20,  # Probability of a 3-legged actuator failure episode during training
        history_len=10,         # RMA proprioceptive history window length (10 steps = 0.20s)
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

        # 1. Multi-Gait Controller State (Walk=0, Trot=1, Bound=2)
        self.base_gait_freq = gait_freq
        self.gait_freq = gait_freq
        self.gait_mode = self.GAIT_TROT
        self.forced_gait_mode = None  # Optional manual override for demos/interactive viewer
        self.gait_phase = 0.0
        self.phase_table = {
            self.GAIT_WALK: np.array([0.0, np.pi, 0.5 * np.pi, 1.5 * np.pi], dtype=np.float32),
            self.GAIT_TROT: np.array([0.0, np.pi, np.pi, 0.0], dtype=np.float32),
            self.GAIT_BOUND: np.array([0.0, 0.0, np.pi, np.pi], dtype=np.float32),
        }
        self.leg_phase_offsets = self.phase_table[self.GAIT_TROT].copy()

        # 2. 3-Legged Fault-Tolerant Limp-Mode State
        self.leg_failure_prob = float(leg_failure_prob)
        self.leg_health = np.ones(4, dtype=np.float32)  # [FR, FL, RR, RL], 1.0=healthy, 0.0=broken
        self.disabled_leg_idx = -1                      # -1 means all 4 legs healthy
        self.fault_trigger_step = -1

        # 3. Proprioceptive History / RMA Buffer (10 steps of joint error & velocity)
        self.history_len = int(history_len)
        self.joint_err_history = deque(maxlen=self.history_len)
        self.joint_vel_history = deque(maxlen=self.history_len)

        # Locate mocap IDs
        bait_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "bait")
        self.bait_mocap_id = int(self.model.body_mocapid[bait_body_id]) if (bait_body_id >= 0 and self.model.body_mocapid[bait_body_id] >= 0) else -1

        fault_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "fault_marker")
        self.fault_mocap_id = int(self.model.body_mocapid[fault_body_id]) if (fault_body_id >= 0 and self.model.body_mocapid[fault_body_id] >= 0) else -1

        # Body IDs and nominal mass
        self.trunk_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "trunk")
        self.nominal_trunk_mass = float(self.model.body_mass[self.trunk_body_id])
        self.hip_body_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in ["FR_hip", "FL_hip", "RR_hip", "RL_hip"]
        ]

        # Actuator and joint setup (12 DoF)
        self.num_joints = 12
        self.action_scale = np.array([
            0.20, 0.28, 0.28,
            0.20, 0.28, 0.28,
            0.20, 0.28, 0.28,
            0.20, 0.28, 0.28
        ], dtype=np.float32)

        # Nominal standing joint angles for Unitree A1
        self.default_qpos = np.array([
            0.0, 0.9, -1.8,
            0.0, 0.9, -1.8,
            0.0, 0.9, -1.8,
            0.0, 0.9, -1.8
        ], dtype=np.float32)

        # Action Space: 12 residual joint angle adjustments
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.num_joints,), dtype=np.float32
        )

        # Observation Space (78 dims):
        # - First 47 dims: Core proprioception (100% aligned with original 47-dim policy for weight transfer!)
        #   3 (base ang vel) + 3 (proj gravity) + 3 (command) + 2 (clock) + 12 (q_err) + 12 (dq) + 12 (last_action)
        # - Next 31 dims: Multi-Gait + Leg Health + RMA Proprioceptive History Summary
        #   3 (gait_mode one-hot: Walk, Trot, Bound)
        #   + 4 (leg_health mask: FR, FL, RR, RL)
        #   + 12 (RMA 10-step temporal joint position error encoding)
        #   + 12 (RMA 10-step temporal joint velocity / slip encoding)
        self.core_obs_dim = 47
        self.obs_dim = 47 + 3 + 4 + 12 + 12  # 78 dims
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
        )

        # State buffers
        self.last_action = np.zeros(self.num_joints, dtype=np.float32)
        self.smoothed_action = np.zeros(self.num_joints, dtype=np.float32)
        self.command = np.array([0.8, 0.0, 0.0], dtype=np.float32)
        self.step_count = 0
        self.max_episode_steps = 1000
        self.prev_x_pos = 0.0

        # Rendering
        self.render_mode = render_mode
        self.camera = None

    def set_max_push_force(self, force: float):
        """Update maximum push force dynamically for curriculum training."""
        self.max_push_force = float(force)

    def set_leg_failure_prob(self, prob: float):
        """Update leg actuator failure probability dynamically for curriculum training."""
        self.leg_failure_prob = float(np.clip(prob, 0.0, 1.0))

    def set_disabled_leg(self, leg_idx: int):
        """
        Manually break or heal a leg at runtime (0=FR, 1=FL, 2=RR, 3=RL, -1=Heal all 4 legs).
        """
        self.disabled_leg_idx = int(leg_idx)
        self.leg_health[:] = 1.0
        if 0 <= self.disabled_leg_idx < 4:
            self.leg_health[self.disabled_leg_idx] = 0.0

    def set_gait_mode(self, mode):
        """
        Manually force a specific gait mode (0=WALK, 1=TROT, 2=BOUND, or None=Auto Bait-Adaptive).
        """
        self.forced_gait_mode = mode

    def _get_robot_yaw(self) -> float:
        """Compute robot trunk yaw angle (rad) in world frame from quaternion."""
        w, x, y, z = self.data.qpos[3:7]
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return float(np.arctan2(siny_cosp, cosy_cosp))

    def _spawn_random_bait(self):
        """Place the bait at the center of a random floor grid cell (360 degrees around the robot)."""
        robot_x, robot_y = float(self.data.qpos[0]), float(self.data.qpos[1])

        angle = np.random.uniform(-np.pi, np.pi)
        radius = np.random.uniform(1.5, 3.4)

        raw_x = robot_x + radius * np.cos(angle)
        raw_y = robot_y + radius * np.sin(angle)

        cell_x = np.round(raw_x / self.cell_size) * self.cell_size
        cell_y = np.round(raw_y / self.cell_size) * self.cell_size

        if np.hypot(cell_x - robot_x, cell_y - robot_y) < 1.2:
            cell_x += self.cell_size * (1.0 if np.cos(angle) >= 0 else -1.0)
            cell_y += self.cell_size * (1.0 if np.sin(angle) >= 0 else -1.0)

        self.bait_pos = np.array([cell_x, cell_y], dtype=np.float32)
        self.prev_bait_dist = float(np.hypot(self.bait_pos[0] - robot_x, self.bait_pos[1] - robot_y))

        if self.bait_mocap_id >= 0:
            self.data.mocap_pos[self.bait_mocap_id] = [self.bait_pos[0], self.bait_pos[1], 0.0]

    def _update_bait_and_gait_schedule(self):
        """
        1. Compute 360-degree steering command [vx, vy, wz] toward the target bait cell.
        2. Automatically select the optimal Gait Mode (Walk / Trot / Bound) based on
           bait distance, heading alignment, and leg health.
        """
        robot_x, robot_y = float(self.data.qpos[0]), float(self.data.qpos[1])
        dx = float(self.bait_pos[0] - robot_x)
        dy = float(self.bait_pos[1] - robot_y)
        dist = float(np.hypot(dx, dy))

        if self.use_bait:
            target_yaw = float(np.arctan2(dy, dx))
            robot_yaw = self._get_robot_yaw()
            yaw_err = (target_yaw - robot_yaw + np.pi) % (2.0 * np.pi) - np.pi

            # Automatic Multi-Gait Switching based on Bait Distance & Heading Error
            if self.forced_gait_mode is not None:
                self.gait_mode = int(self.forced_gait_mode)
            elif self.disabled_leg_idx >= 0:
                # In 3-legged limp mode, use rhythmic tripod trot/hop
                self.gait_mode = self.GAIT_TROT
            elif abs(yaw_err) > 0.75 or dist < 0.95:
                # Sharp turn (>43 deg) or precision approach into bait cell -> 4-Beat Walk
                self.gait_mode = self.GAIT_WALK
            elif dist > 2.25 and abs(yaw_err) < 0.35:
                # Long straightaway toward distant bait -> High-Speed Bound/Gallop
                self.gait_mode = self.GAIT_BOUND
            else:
                # Standard cruising -> Diagonal Trot
                self.gait_mode = self.GAIT_TROT

            # Speed & steering command tailored to active gait mode and leg health
            max_vx = {self.GAIT_WALK: 0.55, self.GAIT_TROT: 0.88, self.GAIT_BOUND: 1.18}[self.gait_mode]
            if self.disabled_leg_idx >= 0:
                max_vx = 0.65  # Safe tripod hopping speed when 1 leg is broken

            wz_cmd = float(np.clip(1.8 * yaw_err, -1.2, 1.2))
            alignment = max(0.20, float(np.cos(yaw_err)))
            desired_speed = float(np.clip(max_vx * min(1.0, dist / 0.5), 0.30, max_vx))
            vx_cmd = desired_speed * alignment
            vy_cmd = float(np.clip(0.25 * np.sin(yaw_err), -0.20, 0.20))

            self.command = np.array([vx_cmd, vy_cmd, wz_cmd], dtype=np.float32)
        else:
            if self.forced_gait_mode is not None:
                self.gait_mode = int(self.forced_gait_mode)

        # Adjust gait frequency and phase offsets for Walk / Trot / Bound / 3-Legged Tripod
        if self.disabled_leg_idx >= 0:
            self.gait_freq = 2.5  # Faster cadence for 3-legged dynamic hopping stability
            # 3-legged tripod coordination: healthy legs spaced smoothly
            self.leg_phase_offsets = self.phase_table[self.GAIT_TROT].copy()
        else:
            if self.gait_mode == self.GAIT_WALK:
                self.gait_freq = 1.85
            elif self.gait_mode == self.GAIT_BOUND:
                self.gait_freq = 2.65
            else:
                self.gait_freq = self.base_gait_freq
            self.leg_phase_offsets = self.phase_table[self.gait_mode].copy()

    def _get_gait_reference(self):
        """
        Compute kinematic reference joint angles for:
          - 3-in-1 Multi-Gait (Walk / Trot / Bound)
          - 360-Degree Sharp Turning
          - 3-Legged Fault-Tolerant Limp Mode (tucks broken leg & shifts CoM support)
        """
        q_ref = self.default_qpos.copy()
        vx_cmd = float(self.command[0])
        vy_cmd = float(self.command[1])
        wz_cmd = float(self.command[2])

        speed_scale = np.clip(abs(vx_cmd) / 0.8, 0.25, 1.45)
        turn_activity = np.clip(abs(wz_cmd) / 1.0, 0.0, 1.0)
        step_activity = max(speed_scale, 0.75 * turn_activity)
        direction = 1.0 if vx_cmd >= 0.0 else -1.0

        # Gait-specific stride & clearance tuning
        if self.gait_mode == self.GAIT_WALK:
            swing_amp = 0.20 * speed_scale * direction
            lift_amp = 0.22 * np.clip(step_activity + 0.3, 0.7, 1.1)
        elif self.gait_mode == self.GAIT_BOUND:
            swing_amp = 0.30 * speed_scale * direction
            lift_amp = 0.27 * np.clip(step_activity + 0.3, 0.8, 1.2)
        else:  # GAIT_TROT
            swing_amp = 0.26 * speed_scale * direction
            lift_amp = 0.24 * np.clip(step_activity + 0.3, 0.7, 1.15)

        hip_lat_amp = 0.09 * np.clip(vy_cmd / 0.25, -1.0, 1.0)

        for leg_idx in range(4):
            base_idx = leg_idx * 3

            # If this leg is broken/disabled, lock it into a tucked-up pose clear of the floor
            if leg_idx == self.disabled_leg_idx:
                q_ref[base_idx + 0] = 0.0
                q_ref[base_idx + 1] = 1.35   # tucked thigh
                q_ref[base_idx + 2] = -2.45  # fully flexed calf (foot lifted high)
                continue

            theta = (self.gait_phase + self.leg_phase_offsets[leg_idx]) % (2.0 * np.pi)
            swing_signal = max(0.0, np.sin(theta))
            stance_press = min(0.0, np.sin(theta))

            is_left_leg = (leg_idx in [1, 3])
            is_front_leg = (leg_idx in [0, 1])
            yaw_stride_mod = (0.15 * wz_cmd) if not is_left_leg else (-0.15 * wz_cmd)
            hip_turn_mod = (0.07 * wz_cmd) if is_front_leg else (-0.07 * wz_cmd)

            # 3-Legged Limp Compensation:
            # If the partner leg on the same front/rear pair is broken, shift the surviving leg's
            # hip abduction slightly inward toward the body centerline to support the missing corner!
            limp_hip_bias = 0.0
            limp_pitch_bias = 0.0
            if self.disabled_leg_idx >= 0:
                broken_is_front = (self.disabled_leg_idx in [0, 1])
                if is_front_leg == broken_is_front:
                    # Surviving partner leg braces inward toward center of mass
                    limp_hip_bias = -0.10 if is_left_leg else 0.10
                    limp_pitch_bias = -0.05
                else:
                    # Opposite pair carries more vertical load
                    limp_pitch_bias = 0.04

            thigh_offset = (swing_amp + yaw_stride_mod) * np.cos(theta) + lift_amp * swing_signal + limp_pitch_bias
            calf_offset = -2.0 * lift_amp * swing_signal + 0.05 * stance_press
            hip_offset = (hip_lat_amp + hip_turn_mod) * np.cos(theta) + limp_hip_bias

            q_ref[base_idx + 0] += hip_offset
            q_ref[base_idx + 1] += thigh_offset
            q_ref[base_idx + 2] += calf_offset

        return q_ref

    def _update_fault_marker(self):
        """Position the visual magenta fault beacon above the disabled leg's hip."""
        if self.fault_mocap_id < 0:
            return
        if 0 <= self.disabled_leg_idx < 4:
            hip_id = self.hip_body_ids[self.disabled_leg_idx]
            hip_pos = self.data.xpos[hip_id]
            self.data.mocap_pos[self.fault_mocap_id] = [hip_pos[0], hip_pos[1], hip_pos[2]]
        else:
            self.data.mocap_pos[self.fault_mocap_id] = [100.0, 100.0, -5.0]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        self.gait_phase = np.random.uniform(0.0, 2.0 * np.pi)
        self.baits_collected = 0

        # Decide if a leg failure will occur in this episode (during training)
        self.set_disabled_leg(-1)
        if np.random.rand() < self.leg_failure_prob:
            # Trigger mid-episode leg failure between step 60 and 300 so policy experiences the transition
            self.fault_trigger_step = int(np.random.randint(60, 300))
        else:
            self.fault_trigger_step = -1

        # Initial robot pose at origin
        self.data.qpos[0:2] = 0.0
        self.data.qpos[2] = 0.28
        self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]

        if self.use_bait:
            self._spawn_random_bait()
            self._update_bait_and_gait_schedule()
        else:
            self.gait_mode = int(self.forced_gait_mode) if self.forced_gait_mode is not None else int(np.random.choice([0, 1, 2]))
            self._update_bait_and_gait_schedule()
            self.command = np.array([
                np.random.uniform(0.5, 1.1),
                np.random.uniform(-0.15, 0.15),
                np.random.uniform(-0.8, 0.8)
            ], dtype=np.float32)
            if self.bait_mocap_id >= 0:
                self.data.mocap_pos[self.bait_mocap_id] = [100.0, 100.0, -5.0]

        q_init = self._get_gait_reference()
        self.data.qpos[7:19] = q_init + np.random.uniform(-0.02, 0.02, size=12)
        self.data.qvel[:] = 0.0
        self.data.qvel[0] = 0.25

        # Domain Randomization: Trunk mass & Ground friction (wide range for RMA adaptation)
        self.model.body_mass[self.trunk_body_id] = max(
            2.0, self.nominal_trunk_mass + np.random.uniform(-0.8, 1.5)
        )
        self.model.geom_friction[:, 0] = np.random.uniform(0.45, 1.35)

        mujoco.mj_forward(self.model, self.data)
        self._update_fault_marker()

        # Reset state & RMA history buffers
        self.last_action[:] = 0.0
        self.smoothed_action[:] = 0.0
        self.push_counter = 0
        self.current_push_force[:] = 0.0
        self.step_count = 0
        self.prev_x_pos = float(self.data.qpos[0])

        self.joint_err_history.clear()
        self.joint_vel_history.clear()
        for _ in range(self.history_len):
            self.joint_err_history.append(np.zeros(self.num_joints, dtype=np.float32))
            self.joint_vel_history.append(np.zeros(self.num_joints, dtype=np.float32))

        return self._get_obs(q_init), {}

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

        # Trigger scheduled mid-episode leg failure if applicable
        if self.fault_trigger_step > 0 and self.step_count == self.fault_trigger_step and self.disabled_leg_idx < 0:
            self.set_disabled_leg(int(np.random.randint(0, 4)))

        # Update closed-loop bait steering and multi-gait selection
        self._update_bait_and_gait_schedule()

        # Low-pass filter residual actions
        self.smoothed_action = 0.65 * action + 0.35 * self.smoothed_action

        # Zero out RL residual on the disabled leg so it stays cleanly tucked
        effective_residual = self.smoothed_action * self.action_scale
        if 0 <= self.disabled_leg_idx < 4:
            b_idx = self.disabled_leg_idx * 3
            effective_residual[b_idx : b_idx + 3] = 0.0

        # Advance gait clock
        self.gait_phase = (self.gait_phase + 2.0 * np.pi * self.gait_freq * self.control_dt) % (2.0 * np.pi)
        q_ref = self._get_gait_reference()

        target_qpos = q_ref + effective_residual
        self.data.ctrl[:12] = target_qpos

        # Apply perturbation force
        self._apply_perturbation()

        applied_torques = []
        for _ in range(self.sim_substeps):
            mujoco.mj_step(self.model, self.data)
            applied_torques.append(self.data.actuator_force[:12].copy())

        self._update_fault_marker()

        # Update RMA proprioceptive history window before computing observation
        joint_pos_err = (self.data.qpos[7:19] - q_ref).astype(np.float32)
        joint_vel_scaled = (self.data.qvel[6:18] * 0.1).astype(np.float32)
        self.joint_err_history.append(joint_pos_err)
        self.joint_vel_history.append(joint_vel_scaled)

        obs = self._get_obs(q_ref)
        reward, info = self._compute_reward(action, q_ref, np.mean(applied_torques, axis=0))

        trunk_height = float(self.data.qpos[2])
        projected_gravity = obs[3:6]

        terminated = False
        if trunk_height < 0.15 or trunk_height > 0.46:
            terminated = True
        if projected_gravity[2] > -0.42:  # tilted more than ~65 deg
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
        info["gait_mode"] = self.GAIT_NAMES.get(self.gait_mode, "TROT")
        info["disabled_leg"] = self.LEG_NAMES.get(self.disabled_leg_idx, "NONE")

        return obs, float(reward), terminated, truncated, info

    def _get_obs(self, q_ref=None):
        if q_ref is None:
            q_ref = self._get_gait_reference()

        base_angvel = self.data.sensor("angular-velocity").data if "angular-velocity" in [self.model.sensor(i).name for i in range(self.model.nsensor)] else self.data.qvel[3:6]

        rot_mat = np.zeros(9)
        mujoco.mju_quat2Mat(rot_mat, self.data.qpos[3:7])
        rot_mat = rot_mat.reshape((3, 3))
        proj_gravity = rot_mat.T @ np.array([0.0, 0.0, -1.0])

        clock = np.array([np.sin(self.gait_phase), np.cos(self.gait_phase)], dtype=np.float32)

        joint_pos_err = (self.data.qpos[7:19] - q_ref).astype(np.float32)
        joint_vel = (self.data.qvel[6:18] * 0.1).astype(np.float32)

        # 1. One-hot Gait Mode vector (3 dims: [is_walk, is_trot, is_bound])
        gait_one_hot = np.zeros(3, dtype=np.float32)
        gait_one_hot[int(self.gait_mode) % 3] = 1.0

        # 2. Leg Health Mask (4 dims: [FR, FL, RR, RL])
        leg_health_vec = self.leg_health.copy()

        # 3. RMA Proprioceptive History Temporal Features (24 dims):
        #    Exponentially weighted mean over the 10-step sliding window captures ground slip,
        #    payload compression, and external push drift online without privileged sensors.
        err_hist = np.array(self.joint_err_history, dtype=np.float32)  # (H, 12)
        vel_hist = np.array(self.joint_vel_history, dtype=np.float32)  # (H, 12)
        weights = np.linspace(0.5, 1.5, len(err_hist), dtype=np.float32)[:, None]
        rma_err_encoding = np.mean(err_hist * weights, axis=0)
        rma_vel_encoding = np.mean(vel_hist * weights, axis=0)

        obs = np.concatenate([
            # Core 47 dims (matches original checkpoint ordering)
            base_angvel,
            proj_gravity,
            self.command,
            clock,
            joint_pos_err,
            joint_vel,
            self.last_action,
            # Extended 31 dims (Multi-Gait + Leg Health + RMA History)
            gait_one_hot,
            leg_health_vec,
            rma_err_encoding,
            rma_vel_encoding,
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

        # 1. Forward velocity tracking reward
        vel_err = (local_linvel[0] - self.command[0]) ** 2
        r_vel = np.exp(-vel_err / 0.22)

        # 2. Progress toward target bait cell
        bait_reached = False
        if self.use_bait:
            robot_x, robot_y = float(self.data.qpos[0]), float(self.data.qpos[1])
            curr_bait_dist = float(np.hypot(self.bait_pos[0] - robot_x, self.bait_pos[1] - robot_y))
            d_bait = self.prev_bait_dist - curr_bait_dist
            self.prev_bait_dist = curr_bait_dist

            r_forward = 2.5 * np.clip(local_linvel[0], -0.2, self.command[0] * 1.2) + 18.0 * d_bait

            if curr_bait_dist <= self.bait_reach_radius:
                bait_reached = True
                self.baits_collected += 1
                # Extra bonus if bait was collected on 3 legs!
                bonus = 35.0 if self.disabled_leg_idx >= 0 else 25.0
                r_forward += bonus
                self._spawn_random_bait()
        else:
            dx = float(self.data.qpos[0]) - self.prev_x_pos
            r_forward = 2.5 * np.clip(local_linvel[0], -0.3, self.command[0] * 1.2) + 15.0 * dx

        # 3. Multi-Gait & 3-Legged Reference Tracking Reward
        gait_err = np.sum(np.square(self.data.qpos[7:19] - q_ref))
        r_gait = np.exp(-gait_err / 0.50)

        # 4. Lateral velocity & vertical bounce penalties
        r_lateral = -1.5 * ((local_linvel[1] - self.command[1]) ** 2) - 0.7 * (base_linvel[2] ** 2)

        # 5. Sharp-turn angular yaw rate tracking
        yaw_err = (local_angvel[2] - self.command[2]) ** 2
        r_yaw = np.exp(-yaw_err / 0.30)
        r_angvel_xy = -0.05 * np.sum(np.square(local_angvel[:2]))

        # 6. Base posture stability (allow slight compensating tilt when 1 leg is broken)
        orient_weight = 1.8 if self.disabled_leg_idx >= 0 else 2.5
        r_orient = -orient_weight * np.sum(np.square(proj_gravity[:2]))

        # 7. Height maintenance around nominal 0.28m
        height_err = max(0.0, abs(self.data.qpos[2] - 0.28) - 0.03)
        r_height = -8.0 * (height_err ** 2)

        # 8. Residual regularization & smoothness
        r_smooth = -0.01 * np.sum(np.square(action - self.last_action)) - 0.01 * np.sum(np.square(action))
        r_torque = -0.00003 * np.sum(np.square(torques))

        # 9. Survival bonus (boosted in 3-legged limp mode to encourage staying upright)
        r_alive = 0.35 if self.disabled_leg_idx >= 0 else 0.20

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
