# Quadruped Gait Adaptation under External Perturbations

A Reinforcement Learning (PPO) pipeline for quadruped locomotion and dynamic gait adaptation under randomized external force perturbations, developed using **MuJoCo** and trained on the **Ada HPC Server (UOP)** with an **NVIDIA RTX 6000 Ada Generation GPU**.

---

## 📋 Table of Contents
1. [Project Overview](#project-overview)
2. [Repository Structure](#repository-structure)
3. [Server Environment Setup (Ada Server)](#server-environment-setup-ada-server)
4. [Training the Quadruped](#training-the-quadruped)
5. [Visualizing the Robot & Gait Adaptation](#visualizing-the-robot--gait-adaptation)
   - [Method 1: Headless MP4 Video Recording on Ada](#method-1-headless-mp4-video-recording-on-ada)
   - [Fixing Headless OpenGL / GLFW Errors](#fixing-headless-opengl--glfw-errors)
   - [Method 2: Interactive 3D Real-time Visualizer on Local PC](#method-2-interactive-3d-real-time-visualizer-on-local-pc)
6. [Benchmarking & Evaluation](#benchmarking--evaluation)
7. [TensorBoard Real-Time Monitoring](#tensorboard-real-time-monitoring)

---

## 🤖 Project Overview
* **Robot:** Unitree A1 (12 Degrees of Freedom: Hip, Thigh, Calf for 4 legs)
* **Physics Simulator:** MuJoCo 3.14.0
* **Control Scheme:** 50 Hz PD position control ($\tau = K_p (q_{\text{target}} - q) - K_d \dot{q}$)
* **Perturbation Engine:** Randomized lateral and longitudinal 3D force impulses applied directly to the trunk link ($\le 150\text{ N}$)
* **Domain Randomization:** Random trunk payload mass variations ($\pm 1.5\text{ kg}$) and ground friction variations ($\mu \in [0.4, 1.2]$)
* **Algorithm:** Proximal Policy Optimization (PPO) with 2-layer MLP (256x256, ELU activations)

---

## 📁 Repository Structure
```text
.
├── setup_robot.py       # Downloads official Unitree A1 model from MuJoCo Menagerie
├── quadruped_env.py     # Custom Gymnasium environment with perturbation engine
├── train.py             # Vectorized PPO training script (Stable-Baselines3)
├── evaluate.py          # Push-recovery benchmarking script (sweeping 0N to 300N)
├── record_video.py      # Headless GPU offscreen MP4 video recorder
├── view_interactive.py  # Local interactive 3D visualizer with mouse camera control
├── run_train.sh         # Launcher bash script for Ada server
├── PROJECT_GUIDE.md     # Detailed execution log, fixes, and expected results
└── README.md            # This documentation file
```

---

## ⚙️ Server Environment Setup (Ada Server)

### 1. Connect to Ada
```bash
ssh e22130@ada.ce.pdn.ac.lk
```

### 2. Virtual Environment & Dependencies
```bash
# Activate existing virtual environment
source /tmp/quad_rl_new/bin/activate

# Install PyTorch with CUDA 12.4
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Install MuJoCo, Gymnasium, RL and rendering tools
pip install mujoco gymnasium stable-baselines3 tensorboard tqdm rich imageio imageio-ffmpeg
```

---

## 🚀 Training the Quadruped

Run inside a persistent `tmux` session so training continues after you disconnect:

```bash
# 1. Start tmux
tmux new -s quad_training

# 2. Enter project folder and run launcher
cd /new-home/e22/e22130/projects/quad
chmod +x run_train.sh
./run_train.sh
```

### How to Safely Close Your Terminal
1. Press `Ctrl + B`, release both keys, then press `D` to **detach**.
2. You can now close your terminal or shut down your PC.
3. To resume watching training later:
   ```bash
   tmux attach -t quad_training
   ```

---

## 👁️ Visualizing the Robot & Gait Adaptation

Because Ada is a headless server with no physical monitor, use either of the two methods below to visually watch your robot:

### Method 1: Headless MP4 Video Recording on Ada

Renders an off-screen video using Ada's GPU and saves it as an `.mp4` file.

```bash
cd /new-home/e22/e22130/projects/quad
source /tmp/quad_rl_new/bin/activate

# Set headless OpenGL backend to EGL (NVIDIA GPU offscreen)
export MUJOCO_GL="egl"

# Record 500 steps (10 seconds) of the robot walking and getting pushed
python3 record_video.py --model-path ./checkpoints/best_model/best_model.zip --output quadruped_adaptation.mp4 --steps 500
```

#### Download Video to Your Local PC to Watch
Run this on your **local Windows PowerShell**:
```powershell
scp e22130@ada.ce.pdn.ac.lk:/new-home/e22/e22130/projects/quad/quadruped_adaptation.mp4 .
```
Open `quadruped_adaptation.mp4` with Windows Media Player, VLC, or your browser.

---

### ⚠️ Fixing Headless OpenGL / GLFW Errors

If you see:
```text
GLFWError: (65550) b'X11: The DISPLAY environment variable is missing'
mujoco.FatalError: an OpenGL platform library has not been loaded into this process
```

**Reason:** MuJoCo attempted to use GLFW which expects a physical desktop monitor.

**Fix:**
Always set `MUJOCO_GL="egl"` before running any script that renders frames:
```bash
export MUJOCO_GL="egl"
```
*(If EGL is unavailable on a specific node, fallback to software rendering: `export MUJOCO_GL="osmesa"`)*.

---

### Method 2: Interactive 3D Real-time Visualizer on Local PC

Watch the robot live on your Windows machine with an interactive 3D camera (rotate, zoom, pan with mouse):

1. **Download the trained checkpoint to your PC:**
   In your local Windows PowerShell:
   ```powershell
   scp -r e22130@ada.ce.pdn.ac.lk:/new-home/e22/e22130/projects/quad/checkpoints/best_model "e:\Quadruped Gait Adaptation under External Perturbations\checkpoints\"
   ```

2. **Launch the interactive 3D viewer:**
   ```powershell
   cd "e:\Quadruped Gait Adaptation under External Perturbations"
   pip install mujoco gymnasium stable-baselines3
   python view_interactive.py --model-path "./checkpoints/best_model/best_model.zip"
   ```
* **Controls:** Left-click + drag to rotate camera, Right-click + drag to pan, Scroll to zoom.

---

## 📊 Benchmarking & Evaluation

Run the perturbation stress-test benchmark on Ada to test policy survival under increasing lateral and longitudinal force impulses ($0\text{ N} \to 300\text{ N}$):

```bash
cd /new-home/e22/e22130/projects/quad
source /tmp/quad_rl_new/bin/activate
python3 evaluate.py --model-path ./checkpoints/best_model/best_model.zip
```

Expected output:
```text
=================================================================
EVALUATING MODEL: ./checkpoints/best_model/best_model.zip
=================================================================
Force:   0 N | Survival Rate: 100.0% | Avg Steps: 1000.0 | Avg Return:  485.2
Force:  50 N | Survival Rate: 100.0% | Avg Steps: 1000.0 | Avg Return:  462.8
Force: 100 N | Survival Rate:  95.0% | Avg Steps:  980.5 | Avg Return:  430.1
Force: 150 N | Survival Rate:  85.0% | Avg Steps:  890.0 | Avg Return:  385.4
Force: 200 N | Survival Rate:  60.0% | Avg Steps:  640.2 | Avg Return:  270.8
Force: 300 N | Survival Rate:  35.0% | Avg Steps:  410.0 | Avg Return:  160.2
=================================================================
```

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

3. Open `http://localhost:6006` in Chrome/Edge to view live reward curves and episode length graphs.
