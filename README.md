# Quadruped Gait Adaptation under External Perturbations

A Reinforcement Learning (PPO) pipeline for **Unitree A1** quadruped locomotion and dynamic push-recovery adaptation under randomized external force perturbations, developed using **MuJoCo 3.14.0** and trained on the **Ada HPC Server (UOP)** with **NVIDIA RTX 6000 Ada Generation GPUs**.

---

## 📋 Table of Contents
1. [Project Overview & Architecture](#project-overview--architecture)
2. [Development & Experiment Log (Yesterday & Today)](#development--experiment-log-yesterday--today)
3. [Training Results & Push-Recovery Benchmark](#training-results--push-recovery-benchmark)
4. [Baseline Test Video vs. RL-Developed Video](#baseline-test-video-vs-rl-developed-video)
5. [Repository Structure](#repository-structure)
6. [Server Environment Setup (Ada Server)](#server-environment-setup-ada-server)
7. [Training the Quadruped](#training-the-quadruped)
8. [Benchmarking & Evaluation](#benchmarking--evaluation)
9. [Visualizing the Robot & Gait Adaptation](#visualizing-the-robot--gait-adaptation)
   - [Method 1: Headless MP4 Video Recording on Ada](#method-1-headless-mp4-video-recording-on-ada)
   - [Fixing Headless OpenGL / GLFW Errors](#fixing-headless-opengl--glfw-errors)
   - [Method 2: Interactive 3D Real-time Visualizer on Local PC](#method-2-interactive-3d-real-time-visualizer-on-local-pc)
10. [TensorBoard Real-Time Monitoring](#tensorboard-real-time-monitoring)

---

## 🤖 Project Overview & Architecture
* **Robot:** Unitree A1 (12 Degrees of Freedom: Hip Abduction, Thigh, and Calf joints across 4 legs; ~12 kg total mass).
* **Physics Simulator:** MuJoCo 3.14.0 (`500 Hz` physics timestep, `50 Hz` control policy loop with 10 substeps).
* **Hybrid Residual Trotting Control Scheme:**
  * **Kinematic Reference Generator (`2.2 Hz` Diagonal Trot):** Computes cyclic diagonal gait reference trajectories (`FR+RL` in phase `0`, `FL+RR` in phase $\pi$) with forward thigh sweep and swing-phase foot clearance.
  * **RL Residual Policy (`12 DoF`):** Outputs smoothed residual joint angle adjustments ($\Delta q$) scaled by `[0.15, 0.25, 0.25]` rad per `[hip, thigh, calf]` on top of the reference gait:
    $$q_{\text{target}} = q_{\text{ref}}(t) + \alpha \cdot \tilde{a}_t, \quad \tilde{a}_t = 0.65 a_t + 0.35 \tilde{a}_{t-1}$$
  * **Actuation:** MuJoCo position actuators (`kp = 100`) tracking $q_{\text{target}}$.
* **Observation Space (`47 dims`):**
  * Base angular velocity (`3`), projected gravity vector in local frame (`3`), velocity command $[v_x, v_y, \omega_z]$ (`3`), gait phase clock $[\sin\phi, \cos\phi]$ (`2`), joint position tracking error $q - q_{\text{ref}}$ (`12`), scaled joint velocities (`12`), and previous actions (`12`).
* **Perturbation Engine & Curriculum (`PushCurriculumCallback`):**
  * Automatically warms up locomotion at `0 N` for the first `300k` steps, then linearly ramps up 3D external trunk impulse forces (`0 N` $\to$ `60 N+`) over `1.5M` steps so the robot masters forward walking before learning aggressive push recovery.
* **Domain Randomization:**
  * Randomized trunk payload mass (`nominal - 0.8 kg` to `nominal + 1.2 kg` without cumulative drift) and ground friction ($\mu \in [0.7, 1.3]$) every episode reset.
* **RL Algorithm:** Proximal Policy Optimization (PPO) via Stable-Baselines3 with a 2-layer MLP (`256 x 256`, `ELU` activations) across 8 parallel `SubprocVecEnv` workers.

---

## 🗓️ Development & Experiment Log (Yesterday & Today)

### Day 1 (Yesterday): Environment Setup, Diagnostics & Gait Formulation
1. **Ada HPC Server Setup:**
   * Verified 3x **NVIDIA RTX 6000 Ada Generation** GPUs (48 GB VRAM, CUDA 13.0 driver) and created the `/tmp/quad_rl_new` Python 3.12 virtual environment.
   * Automated downloading of the Google DeepMind MuJoCo Menagerie `unitree_a1` XML/meshes via `setup_robot.py`.
   * Configured headless GPU rendering using `export MUJOCO_GL="egl"` to resolve headless X11/GLFW display errors on the server.
2. **Addressing Early Training Stagnation (Test Videos v1 & v2):**
   * In initial pure-RL runs (`quadruped_adaptation.mp4`), the policy learned to stand still or take tiny shuffling steps to avoid falling under immediate `150 N` pushes, and actuator control mismatches (`data.ctrl` vs torque) limited locomotion.
   * **Upgrades Implemented:**
     * Built the **2.2 Hz diagonal trotting kinematic reference** (`_get_gait_reference()`) combined with **residual RL control** in `quadruped_env.py`.
     * Corrected Unitree A1 thigh/calf swing kinematics so stance legs sweep front-to-back to propel the torso in `+X`, reaching `~10.8 m` in 12 seconds (`quadruped_adaptation_v2.mp4`).
     * Added **cell-by-cell forward progress rewards** (`15.0 * dx`), an **anti-stagnation penalty**, **low-pass action smoothing**, and **`PushCurriculumCallback`** in `train.py`.

### Day 2 (Today): Full Curriculum PPO Training & Push-Recovery Benchmarking
1. **Completed 3,000,000+ Timestep PPO Training on Ada:**
   * Trained across 8 vectorized environments (`3,014,656` total timesteps in `8,496` seconds at `~354 FPS`).
   * Achieved a perfect **`1000.00 +/- 0.00` evaluation episode length** (zero falls) and **`7,190` mean evaluation reward**.
   * Saved both `./checkpoints/best_model/best_model.zip` and `./checkpoints/final_model.zip`.
2. **Executed Multi-Force Push Recovery Benchmark (`evaluate.py`):**
   * Evaluated the trained policy across external push forces from `0 N` to `120 N` (`20 N` increments).
   * Achieved **`100.0%` survival rate from `0 N` through `100 N`** and **`80.0%` survival at `120 N`**, while consistently walking **`19.48 m – 20.67 m`** per 1000-step episode.
3. **Recorded Final RL-Adapted Rollout Video (`record_video.py`):**
   * Rendered the closed-loop RL policy actively stabilizing its torso and widening foot placement under external perturbations (`quadruped_adaptation_v3.mp4`).

---

## 🏆 Training Results & Push-Recovery Benchmark

### 1. Final PPO Training Metrics (`3.01M` Timesteps)
```text
Episode length: 1000.00 +/- 0.00
-----------------------------------------
| eval/                   |             |
|    mean_ep_length       | 1e+03       |
|    mean_reward          | 7.19e+03    |
| time/                   |             |
|    total_timesteps      | 3000000     |
| train/                  |             |
|    approx_kl            | 0.035052232 |
|    clip_fraction        | 0.408       |
|    clip_range           | 0.2         |
|    entropy_loss         | -13.9       |
|    explained_variance   | 0.959       |
|    learning_rate        | 0.0003      |
|    loss                 | 11.8        |
|    n_updates            | 1830        |
|    policy_gradient_loss | -0.00541    |
|    std                  | 0.774       |
|    value_loss           | 24          |
-----------------------------------------
[SUCCESS] Training completed! Model saved to: ./checkpoints/final_model.zip
```

### 2. Perturbation Stress-Test Benchmark (`evaluate.py`)
Evaluated at a commanded forward velocity of $v_x = 0.85\text{ m/s}$ over `1000` steps (`20.0 s`) per episode:

| External Push Force (N) | Survival Rate (%) | Avg Forward Distance (m) | Avg Episode Steps | Avg Return |
| :---: | :---: | :---: | :---: | :---: |
| **0 N** | **100.0%** | `20.59 m` | `1000.0` | `7178.1` |
| **20 N** | **100.0%** | `20.67 m` | `1000.0` | `7150.9` |
| **40 N** | **100.0%** | `20.65 m` | `1000.0` | `7116.7` |
| **60 N** | **100.0%** | `20.31 m` | `1000.0` | `7057.9` |
| **80 N** | **100.0%** | `20.03 m` | `1000.0` | `7028.9` |
| **100 N** | **100.0%** | `19.81 m` | `1000.0` | `6989.3` |
| **120 N** | **80.0%** | `19.48 m` | `993.8` | `6866.7` |

> **Key Takeaway:** Even under **`100 N` lateral/longitudinal impulses** (~85% of the 12 kg robot's total body weight), the trained PPO policy maintains a **`100.0%` survival rate** and travels **`19.81 m`** (`~0.99 m/s` average forward speed). At **`120 N`** (~10 kgf impact), it still achieves **`80.0%` survival** and **`993.8` average steps**.

---

## 🔍 Baseline Test Video vs. RL-Developed Video

The visual and physical differences between the initial test video (`quadruped_adaptation_v2.mp4`, open-loop reference trot) and the final RL-trained policy video (`quadruped_adaptation_v3.mp4`) highlight how residual RL adapts the gait in real time:

| Aspect | Baseline Test Video (Kinematic Reference Only, `action = 0`) | RL-Developed Video (Trained PPO Residual Policy) |
| :--- | :--- | :--- |
| **Control Mode** | **Open-Loop:** Executes fixed sinusoidal joint trajectories (`_get_gait_reference()`) with zero feedback from IMU or joint sensors. | **Closed-Loop (`50 Hz`):** Reads 47D state observations and injects 12D residual joint corrections (`q_ref + smoothed_action * action_scale`). |
| **Lateral Push Response** | **Passive / Stumbling:** Legs continue swinging along a narrow straight-line path when pushed sideways, causing body roll and tipping. | **Active Lateral Stepping:** Uses hip abduction residuals (`±0.15 rad`) to step outward in the direction of the push and widen the support polygon. |
| **Torso Roll & Pitch Stability** | **Wobbles Under Load:** External impulses induce uncompensated pitch/roll oscillations and height drops. | **Active Posture Damping:** Stance legs apply corrective thigh/calf residuals (`±0.25 rad`) to counter gravity tilt and hold trunk height near `0.28 m`. |
| **Forward Velocity & Distance** | **Drift & Slipping:** Loses forward momentum or drifts off-axis after repeated perturbations (~`10.8 m` in `12 s` unperturbed, falls under strong pushes). | **High-Speed Recovery:** Consistently reaches **`~20.6 m` in `20 s`** unperturbed and **`19.81 m`** under `100 N` pushes while staying aligned with the `+X` track. |

---

## 📁 Repository Structure
```text
.
├── setup_robot.py              # Downloads official Unitree A1 model from MuJoCo Menagerie
├── quadruped_env.py            # Custom Gymnasium environment (2.2Hz trot + 12-DoF RL residuals + pushes)
├── train.py                    # Vectorized PPO training script with PushCurriculumCallback
├── evaluate.py                 # Push-recovery benchmarking script (0 N to 120 N sweep)
├── record_video.py             # Headless EGL GPU offscreen MP4 video recorder
├── view_interactive.py         # Local interactive 3D visualizer with mouse camera control
├── run_train.sh                # Launcher bash script for Ada server
├── quadruped_adaptation.mp4    # Early baseline video (initial exploration)
├── quadruped_adaptation_v2.mp4 # Open-loop kinematic trotting test video
├── PROJECT_GUIDE.md            # Detailed execution log, fixes, and server diagnostics
└── README.md                   # Complete project documentation and benchmark results
```

---

## ⚙️ Server Environment Setup (Ada Server)

### 1. Connect to Ada
```bash
ssh e22130@ada.ce.pdn.ac.lk
```

### 2. Virtual Environment & Dependencies
```bash
# Activate virtual environment
source /tmp/quad_rl_new/bin/activate

# Install PyTorch with CUDA 12.4
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Install MuJoCo, Gymnasium, RL, and video rendering packages
pip install mujoco gymnasium stable-baselines3 tensorboard tqdm rich imageio imageio-ffmpeg
```

---

## 🚀 Training & 360° Bait Fine-Tuning

### 1. Upload Updated Scripts to Ada (from Local Windows PowerShell)
```powershell
cd "e:\Quadruped Gait Adaptation under External Perturbations"
scp quadruped_env.py train.py evaluate.py record_video.py run_train.sh e22130@ada.ce.pdn.ac.lk:/new-home/e22/e22130/projects/quad/
```

### 2. Fine-Tune in 360° Random Cell Bait Mode (on Ada)
Fine-tunes your existing `./checkpoints/best_model/best_model.zip` for `1,000,000` steps so the quadruped pivots sharply ($360^\circ$) toward randomly spawned floor cell baits while rejecting external pushes:

```bash
# 1. Start or attach tmux session
tmux new -s quad_bait

# 2. Run the 360° Bait Fine-Tuning launcher
cd /new-home/e22/e22130/projects/quad
chmod +x run_train.sh
./run_train.sh
```

Or run `train.py` directly:
```bash
export CUDA_VISIBLE_DEVICES=1
export MUJOCO_GL="egl"
python3 train.py --num-envs 8 --push-force 60.0 --total-timesteps 1000000 --resume-from ./checkpoints/best_model/best_model.zip --device auto
```

### How to Safely Detach & Reconnect `tmux`
* **Detach:** Press `Ctrl + B`, release both keys, then press `D`.
* **Reattach later:**
  ```bash
  tmux attach -t quad_bait
  ```

---

## 📊 Benchmarking & Evaluation

Run the perturbation stress-test benchmark on Ada to evaluate policy survival, forward distance, and return across increasing external forces (`0 N` $\to$ `120 N`):

```bash
cd /new-home/e22/e22130/projects/quad
source /tmp/quad_rl_new/bin/activate
python3 evaluate.py --model-path ./checkpoints/best_model/best_model.zip
```

---

## 👁️ Visualizing the Robot & Gait Adaptation

Because Ada is a headless server without a physical monitor, use either of the two methods below to view the robot:

### Method 1: Headless MP4 Video Recording on Ada

Renders an off-screen video using Ada's GPU via `EGL` and saves it as an `.mp4` file:

```bash
cd /new-home/e22/e22130/projects/quad
source /tmp/quad_rl_new/bin/activate

# Set headless OpenGL backend to EGL (NVIDIA GPU offscreen)
export MUJOCO_GL="egl"

# Record 600 steps (12 seconds) of the trained policy walking and recovering from pushes
python3 record_video.py --model-path ./checkpoints/best_model/best_model.zip --output quadruped_adaptation_v3.mp4 --steps 600 --push-force 55.0
```

#### Download Video to Your Local Windows PC
Run this in your **local Windows PowerShell**:
```powershell
cd "e:\Quadruped Gait Adaptation under External Perturbations"
scp e22130@ada.ce.pdn.ac.lk:/new-home/e22/e22130/projects/quad/quadruped_adaptation_v3.mp4 .
```

---

### ⚠️ Fixing Headless OpenGL / GLFW Errors

If you encounter:
```text
GLFWError: (65550) b'X11: The DISPLAY environment variable is missing'
mujoco.FatalError: an OpenGL platform library has not been loaded into this process
```
**Fix:** Export `MUJOCO_GL="egl"` prior to running Python (already automatically configured at the top of `record_video.py`):
```bash
export MUJOCO_GL="egl"
```

---

### Method 2: Interactive 3D Real-time Visualizer on Local PC

Watch the robot live on your Windows PC with an interactive 360° 3D camera:

1. **Download the trained checkpoints to your PC (PowerShell):**
   ```powershell
   mkdir -Force "e:\Quadruped Gait Adaptation under External Perturbations\checkpoints"
   scp -r e22130@ada.ce.pdn.ac.lk:/new-home/e22/e22130/projects/quad/checkpoints/* "e:\Quadruped Gait Adaptation under External Perturbations\checkpoints\"
   ```

2. **Launch the interactive 3D viewer:**
   ```powershell
   cd "e:\Quadruped Gait Adaptation under External Perturbations"
   python view_interactive.py --model-path "./checkpoints/best_model/best_model.zip"
   ```
* **Controls:** Left-click + drag to rotate camera, Right-click + drag to pan, Scroll to zoom.

---

## 📈 TensorBoard Real-Time Monitoring

1. **On Ada Server:**
   ```bash
   source /tmp/quad_rl_new/bin/activate
   cd /new-home/e22/e22130/projects/quad
   tensorboard --logdir logs/tb --port 6006
   ```

2. **On your local Windows PC (SSH Tunnel):**
   ```powershell
   ssh -L 6006:localhost:6006 e22130@ada.ce.pdn.ac.lk
   ```

3. Open `http://localhost:6006` in your browser to inspect training curves.
