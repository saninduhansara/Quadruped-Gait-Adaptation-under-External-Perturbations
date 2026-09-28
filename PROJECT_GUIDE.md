# Quadruped Gait Adaptation under External Perturbations
**Platform:** Ada Server (UOP) — NVIDIA RTX 6000 Ada Generation GPU  
**Simulator:** MuJoCo 3.14.0  
**Algorithm:** Proximal Policy Optimization (PPO) via Stable-Baselines3  
**Robot Model:** Unitree A1 (12 DoF Quadruped from Google DeepMind MuJoCo Menagerie)  
**User:** `e22130`  
**Working Directory:** `/new-home/e22/e22130/projects/quad`

---

## 1. What Has Been Done (Detailed Log & Commands)

### 1.1 Server Diagnostics & Environment Setup
1. **Module System Check:**
   - Attempted: `module load python/3.10 cuda/12.1` $\to$ Returned: `module: command not found`.
   - **Resolution:** Ada is configured with system-wide software rather than Lmod/Environment Modules. Direct python paths and drivers are used.
2. **GPU & Driver Identification:**
   - Ran `nvidia-smi` and identified 3x **NVIDIA RTX 6000 Ada Generation** (48 GB VRAM each).
   - Driver version: `580.82.07`, CUDA Version: `13.0`.
   - Confirmed `nvcc` is not required for PyTorch runtime.
3. **Python Virtual Environment:**
   - Python version: `3.12.3`.
   - Virtual environment path: `/tmp/quad_rl_new`.
   - Installed packages:
     ```bash
     source /tmp/quad_rl_new/bin/activate
     pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
     pip install mujoco gymnasium stable-baselines3 tensorboard tqdm rich
     ```
   - Verified installation:
     ```bash
     python3 -c "import torch, mujoco; print('CUDA Available:', torch.cuda.is_available()); print('MuJoCo Version:', mujoco.__version__)"
     ```
     **Result:** `CUDA Available: True`, `MuJoCo Version: 3.14.0`.
4. **SLURM vs Direct Workstation Execution:**
   - Attempted: `sbatch submit_job.slurm` $\to$ Returned: `Command 'sbatch' not found`.
   - **Resolution:** Ada functions as a standalone multi-GPU server. Long-running jobs are executed directly using `tmux` or `nohup`.

---

### 1.2 Code Files Created

#### File 1: `setup_robot.py`
Downloads the official Unitree A1 model and meshes from Google DeepMind's MuJoCo Menagerie.
```python
"""
Setup script to download the Unitree A1 robot model from DeepMind MuJoCo Menagerie.
"""
import os
import urllib.request
import zipfile
import subprocess
import shutil

def setup_robot_model():
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    os.makedirs(model_dir, exist_ok=True)
    a1_dir = os.path.join(model_dir, "unitree_a1")
    
    if os.path.exists(os.path.join(a1_dir, "a1.xml")):
        print(f"[OK] Unitree A1 model already exists at: {a1_dir}")
        return os.path.join(a1_dir, "a1.xml")
    
    print("[INFO] Cloning Google DeepMind MuJoCo Menagerie (Unitree A1)...")
    cmd = [
        "git", "clone", "--depth", "1", "--filter=blob:none", "--sparse",
        "https://github.com/google-deepmind/mujoco_menagerie.git",
        os.path.join(model_dir, "menagerie_tmp")
    ]
    try:
        subprocess.run(cmd, check=True)
        sparse_cmd = ["git", "-C", os.path.join(model_dir, "menagerie_tmp"), "sparse-checkout", "set", "unitree_a1"]
        subprocess.run(sparse_cmd, check=True)
        shutil.move(os.path.join(model_dir, "menagerie_tmp", "unitree_a1"), a1_dir)
        shutil.rmtree(os.path.join(model_dir, "menagerie_tmp"))
        print(f"[SUCCESS] Unitree A1 model ready at: {a1_dir}")
        return os.path.join(a1_dir, "a1.xml")
    except Exception as e:
        print(f"[FALLBACK] Downloading repository archive: {e}")
        zip_url = "https://github.com/google-deepmind/mujoco_menagerie/archive/refs/heads/main.zip"
        zip_path = os.path.join(model_dir, "menagerie.zip")
        urllib.request.urlretrieve(zip_url, zip_path)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            for member in zip_ref.namelist():
                if "unitree_a1/" in member:
                    zip_ref.extract(member, model_dir)
        extracted_path = os.path.join(model_dir, "mujoco_menagerie-main", "unitree_a1")
        if os.path.exists(extracted_path):
            shutil.move(extracted_path, a1_dir)
            shutil.rmtree(os.path.join(model_dir, "mujoco_menagerie-main"))
        if os.path.exists(zip_path):
            os.remove(zip_path)
        print(f"[SUCCESS] Downloaded Unitree A1 model to: {a1_dir}")
        return os.path.join(a1_dir, "a1.xml")

if __name__ == "__main__":
    setup_robot_model()
```

---

#### File 2: `quadruped_env.py`
Custom Gymnasium environment implementing:
- 12-DoF proportional-derivative (PD) position control: $\tau = K_p (q_{\text{target}} - q) - K_d \dot{q}$.
- **Perturbation Engine**: Injects randomized external 3D force vectors ($\le 150\text{ N}$) directly onto the trunk body for multi-step durations.
- **Domain Randomization**: Randomizes trunk mass ($\pm 1.5\text{ kg}$) and ground friction ($\mu \in [0.4, 1.2]$) at each episode reset.
- **Observation Space (45 dims)**: Base angular velocity, projected gravity vector, target velocity command ($v_x, v_y, \omega_z$), relative joint positions, joint velocities, and previous action history.
- **Reward Function**: Forward tracking, yaw tracking, upright torso stability, height maintenance, energy minimization, and survival bonus.

---

#### File 3: `train.py`
Multi-environment vectorized PPO training pipeline:
- Runs 8 parallel simulation environments using `SubprocVecEnv`.
- Neural network architecture: 2-layer MLP (256x256) with `ELU` activation.
- Automated evaluation callback and checkpoint saving.
- Output directory: `/new-home/e22/e22130/projects/quad/logs` and `./checkpoints`.

---

#### File 4: `evaluate.py`
Benchmarking tool to evaluate policy robustness under increasing external push forces ($0\text{ N} \to 300\text{ N}$):
```bash
python3 evaluate.py --model-path ./checkpoints/best_model/best_model.zip
```

---

#### File 5: `run_train.sh`
Launcher script setting GPU assignment and headless EGL rendering:
```bash
#!/bin/bash
source /tmp/quad_rl_new/bin/activate
export CUDA_VISIBLE_DEVICES=1
export MUJOCO_GL="egl"
cd /new-home/e22/e22130/projects/quad
python3 train.py --num-envs 8 --push-force 150.0 --total-timesteps 5000000 --device auto
```

---

### 1.3 Resolved Issues During Launch
* **Missing progress bar dependencies:** Fixed by running `pip install tqdm rich` and setting `progress_bar=False` in `train.py`.
* **Current Execution Status:** Training is successfully executing on GPU 1 at **~476 FPS** inside `tmux`.

---

## 2. What You Need to Do

### 2.1 Managing the Running Job
* **Detach from tmux (safely close terminal):**
  1. Press `Ctrl + B`.
  2. Release, then press `D`.
  3. Close your SSH connection or terminal.
* **Reconnect and check live training:**
  ```bash
  ssh e22130@ada.ce.pdn.ac.lk
  tmux attach -t quad_training
  ```

### 2.2 Monitoring Training via TensorBoard
Run TensorBoard inside your project folder on Ada:
```bash
source /tmp/quad_rl_new/bin/activate
cd /new-home/e22/e22130/projects/quad
tensorboard --logdir logs/tb --port 6006
```
On your local PC, create an SSH tunnel to view it in your browser:
```bash
ssh -L 6006:localhost:6006 e22130@ada.ce.pdn.ac.lk
```
Open `http://localhost:6006` in your browser to monitor real-time loss, reward, and episode lengths.

### 2.3 Post-Training Evaluation
Once the 5,000,000 steps complete:
1. Run the perturbation stress benchmark:
   ```bash
   python3 evaluate.py --model-path ./checkpoints/best_model/best_model.zip
   ```
2. Verify the model across push forces: $0\text{ N}$, $50\text{ N}$, $100\text{ N}$, $150\text{ N}$, $200\text{ N}$, $300\text{ N}$.

### 2.4 Visualizing the Robot (Headless MP4 & Local 3D Viewer)
#### A. Record MP4 on Ada Server (Headless GPU Rendering)
```bash
source /tmp/quad_rl_new/bin/activate
cd /new-home/e22/e22130/projects/quad
export MUJOCO_GL="egl"
python3 record_video.py --model-path ./checkpoints/best_model/best_model.zip --output quadruped_adaptation.mp4 --steps 500
```
* **Download to your Windows PC to view:**
  ```powershell
  scp e22130@ada.ce.pdn.ac.lk:/new-home/e22/e22130/projects/quad/quadruped_adaptation.mp4 .
  ```

#### B. Troubleshooting Headless OpenGL / GLFW Errors:
If you see:
```text
GLFWError: (65550) b'X11: The DISPLAY environment variable is missing'
mujoco.FatalError: an OpenGL platform library has not been loaded into this process
```
**Fix:** Set `export MUJOCO_GL="egl"` (or `export MUJOCO_GL="osmesa"`) before running the script so MuJoCo bypasses GLFW and renders directly to an off-screen GPU buffer.

#### C. Live Interactive 3D Viewer on Local Windows PC
Download the checkpoint to your PC and run:
```powershell
scp -r e22130@ada.ce.pdn.ac.lk:/new-home/e22/e22130/projects/quad/checkpoints/best_model "e:\Quadruped Gait Adaptation under External Perturbations\checkpoints\"
cd "e:\Quadruped Gait Adaptation under External Perturbations"
python view_interactive.py --model-path "./checkpoints/best_model/best_model.zip"
```


---

## 3. Expected Results & Observations

### 3.1 Training Progression Phases
| Training Stage | Expected Episode Length | Expected Behavior |
| :--- | :--- | :--- |
| **0 – 100k Steps** | $8 - 30$ steps | Robot falls almost immediately while exploring joint limits. |
| **100k – 500k Steps** | $50 - 200$ steps | Robot learns to stand upright and stagger forward. |
| **500k – 2M Steps** | $400 - 800$ steps | Emergence of a stable trotting gait; tracks target velocities. |
| **2M – 5M Steps** | $900 - 1000$ steps | Robust dynamic balance; actively compensates for lateral & longitudinal pushes without tipping. |

### 3.2 Expected Perturbation Recovery Metrics
* **0 N to 100 N Pushes:** $\approx 95\% - 100\%$ survival rate. The robot adjusts its foot placement to absorb the shock and continues forward.
* **150 N Pushes (Training Limit):** $\approx 80\% - 90\%$ survival rate. Noticeable lateral stepping and momentary deceleration during push.
* **200 N to 300 N Pushes (Out-of-Distribution):** Survival rate degrades gracefully ($40\% - 65\%$), showing the physical limit of the 12-DoF actuator torque ($33.5\text{ Nm}$).
