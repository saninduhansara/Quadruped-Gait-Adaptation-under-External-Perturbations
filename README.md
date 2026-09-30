# Quadruped Gait Adaptation under External Perturbations & Actuator Failure Recovery

<p align="center">
  <a href="https://github.com/saninduhansara/Quadruped-Gait-Adaptation-under-External-Perturbations">
    <img src="https://img.shields.io/badge/GitHub-Repository-181717?style=for-the-badge&logo=github" alt="GitHub Repo">
  </a>
  <img src="https://img.shields.io/badge/Simulator-MuJoCo%203.14-blue?style=for-the-badge&logo=openai" alt="MuJoCo">
  <img src="https://img.shields.io/badge/Algorithm-PPO%20(Stable--Baselines3)-brightgreen?style=for-the-badge" alt="PPO">
  <img src="https://img.shields.io/badge/Robot-Unitree%20A1%20(12--DoF)-orange?style=for-the-badge" alt="Unitree A1">
  <img src="https://img.shields.io/badge/License-MIT-purple?style=for-the-badge" alt="License">
</p>

An open-source, research-grade Deep Reinforcement Learning (PPO) framework for the **Unitree A1 (12 DoF)** quadruped robot. The system achieves dynamic **360° Random Cell Bait Navigation**, **3-in-1 Multi-Gait Transitions (Walk $\leftrightarrow$ Trot $\leftrightarrow$ Bound)**, **3-Legged Fault-Tolerant Limp-Mode Recovery (Actuator Failure)**, and **10-Step Rapid Motor Adaptation (RMA)** under severe randomized 3D force perturbations ($\le 120\text{ N}$).

---

## 🎥 Autoplay Video Demonstrations

### 1. Unified Multi-Gait, 3-Legged Limp Mode & Bait Navigation Final Demo (`quadruped_unified_demo.mp4`)
Demonstrates the final trained policy autonomously navigating toward randomly spawned grid cell targets, dynamically shifting gaits (`WALK` $\leftrightarrow$ `TROT` $\leftrightarrow$ `BOUND`), and instantly recovering into **3-Legged Limp Mode** when an actuator failure is injected mid-run:

<p align="center">
  <video src="quadruped_unified_demo.mp4" width="100%" controls autoplay loop muted playsinline></video>
</p>

*Direct file link:* [`quadruped_unified_demo.mp4`](./quadruped_unified_demo.mp4)

---

### 2. Video of Training the Robot in Unified Multi-Gait, 3-Legged Limp Mode & Bait Navigation (`training_progression_2min.mp4`)
A comprehensive 2+ minute chronological progression video capturing the evolutionary training process of the robot across 7 key developmental milestones with a real-time elapsed timer (`MM:SS / 02:06`):
1. **0:00 – 0:18:** Step 0 Untrained Policy (flailing and exploring joint limits)
2. **0:18 – 0:36:** ~200k Steps (learning standing balance and upright posture)
3. **0:36 – 0:54:** ~1.0M Steps (rhythmic trotting gait emerges)
4. **0:54 – 1:12:** 3.0M Steps (surviving severe 75N external pushes)
5. **1:12 – 1:30:** Dynamic Multi-Gait Transitions (`WALK` $\leftrightarrow$ `TROT` $\leftrightarrow$ `BOUND`)
6. **1:30 – 1:48:** 360° Random Cell Bait Hunting across the arena
7. **1:48 – 2:06+:** 3-Legged Limp Mode Recovery (Front-Right leg severed while continuing to catch baits on 3 legs)

<p align="center">
  <video src="training_progression_2min.mp4" width="100%" controls autoplay loop muted playsinline></video>
</p>

*Direct file link:* [`training_progression_2min.mp4`](./training_progression_2min.mp4)

---

## 📋 Table of Contents
1. [Target Problem Statement & Engineering Challenges](#-target-problem-statement--engineering-challenges)
2. [Our Approach: Hierarchical Hybrid RL Paradigm](#-our-approach-hierarchical-hybrid-rl-paradigm)
3. [Deep Reinforcement Learning & Mathematical Formulation](#-deep-reinforcement-learning--mathematical-formulation)
   - [PPO Clipped Surrogate Objective & GAE](#1-ppo-clipped-surrogate-objective--gae)
   - [Analytical Kinematic Reference Trajectories](#2-analytical-kinematic-reference-trajectories)
   - [Residual Action Space & Smoothing Filter](#3-residual-action-space--smoothing-filter)
   - [Closed-Loop 360° Waypoint Steering Dynamics](#4-closed-loop-360-waypoint-steering-dynamics)
   - [3-Legged Limp-Mode Center-of-Mass Compensation](#5-3-legged-limp-mode-center-of-mass-compensation)
   - [RMA 10-Step Proprioceptive History Window](#6-rma-10-step-proprioceptive-history-window)
   - [Multi-Objective Reward Formulation](#7-multi-objective-reward-formulation)
4. [Unified Technical Architecture & State Spaces](#-unified-technical-architecture--state-spaces)
5. [Experimental Results & Benchmark Data](#-experimental-results--benchmark-data)
   - [PPO Training Convergence Metrics](#ppo-training-convergence-metrics)
   - [Force Sweep Benchmark: 4-Legged vs. 3-Legged Limp Mode](#force-sweep-benchmark-4-legged-vs-3-legged-limp-mode)
   - [Ablation: Baseline Open-Loop Trot vs. Closed-Loop RL](#ablation-baseline-open-loop-trot-vs-closed-loop-rl)
6. [Repository Structure](#-repository-structure)
7. [Installation & Setup](#-installation--setup)
8. [Step-by-Step Execution Guide](#-step-by-step-execution-guide)
   - [1. Launching Interactive 3D Real-Time Simulator](#1-launching-interactive-3d-real-time-simulator)
   - [2. Training & Fine-Tuning the Unified Model](#2-training--fine-tuning-the-unified-model)
   - [3. Running Benchmarks (4-Leg & 3-Leg Limp Mode)](#3-running-benchmarks-4-leg--3-leg-limp-mode)
   - [4. Generating 2+ Minute Showcase Videos](#4-generating-2-minute-showcase-videos)
9. [Key Engineering Problems & Solutions](#-key-engineering-problems--solutions)
10. [License & Citation](#-license--citation)

---

## 🎯 Target Problem Statement & Engineering Challenges

Quadrupedal robot locomotion in unstructured, real-world environments faces four fundamental failure modes:

1. **Susceptibility to Lateral Force Impulses:** External disturbances (collisions, crosswinds, sudden physical pushes) inject instantaneous unmodeled momentum into the trunk. Fixed-pattern or open-loop controllers cannot adjust foot placement in real time, leading to tipping.
2. **Catastrophic Actuator Failure:** In real deployments (e.g., search-and-rescue or planetary exploration), joint motors can overheat, strip gears, or sever wiring. Standard quadrupeds cannot survive losing a single leg and immediately collapse onto the unactuated corner.
3. **Friction and Mass Variation (Blind Environments):** Robots frequently transition between high-traction concrete ($\mu \approx 1.2$) and low-traction ice or gravel ($\mu \approx 0.4$) with variable payload weights without explicit friction or mass sensors.
4. **Agile Omnidirectional Goal Navigation:** Reaching arbitrary targets in a 2D plane requires dynamically modulating forward speed, lateral stepping, and yaw turning simultaneously without inducing instability during aggressive pivots.

---

## 💡 Our Approach: Hierarchical Hybrid RL Paradigm

Pure end-to-end Reinforcement Learning from scratch frequently suffers from **standing stagnation** (the agent discovers that standing motionless minimizes fall penalties when exposed to heavy early pushes) and requires millions of sample interactions to discover basic coordination.

To solve this, we formulated a **Hierarchical Hybrid Control Paradigm**:

```
                         ┌─────────────────────────────────────────────────────────┐
                         │           Goal Generator: Random Cell Bait Target       │
                         └────────────────────────────┬────────────────────────────┘
                                                      │
                                                      ▼
                         ┌─────────────────────────────────────────────────────────┐
                         │   High-Level 360° Steering & Adaptive Gait Selector     │
                         │   - Target Yaw Error: Δψ ∈ [-π, π]                      │
                         │   - Gait Selection: WALK (sharp turn) / TROT / BOUND    │
                         └────────────────────────────┬────────────────────────────┘
                                                      │
                                                      ▼
                         ┌─────────────────────────────────────────────────────────┐
                         │      Analytical Kinematic Gait Reference Generator      │
                         │      q_ref(t) for 12 DoF with Stride & Limp Modulation  │
                         └────────────────────────────┬────────────────────────────┘
                                                      │
                       ┌──────────────────────────────┴──────────────────────────────┐
                       │                                                             │
                       ▼                                                             ▼
┌──────────────────────────────────────────────┐              ┌──────────────────────────────────────────────┐
│  Nominal Cyclic Joint Trajectory q_ref(t)   │              │  10-Step RMA Proprioceptive History Window   │
│  (Base Trotting / Walking / Bounding)        │              │  z_err, z_vel ∈ R^24 (Implicit Slip & Mass)  │
└──────────────────────┬───────────────────────┘              └──────────────────────┬───────────────────────┘
                       │                                                             │
                       │                                                             ▼
                       │                                              ┌──────────────────────────────┐
                       │                                              │ Deep Actor-Critic Policy     │
                       │                                              │ π_θ(a_t | s_t), s_t ∈ R^78   │
                       │                                              └──────────────┬───────────────┘
                       │                                                             │
                       ▼                                                             ▼
                       └──────────────────────────────┬──────────────────────────────┘
                                                      │
                                                      ▼
                         ┌─────────────────────────────────────────────────────────┐
                         │   Low-Pass Filtered Residual Joint Scaling & Safety     │
                         │   q_target = q_ref(t) + α ⊙ a_smoothed                  │
                         └────────────────────────────┬────────────────────────────┘
                                                      │
                                                      ▼
                         ┌─────────────────────────────────────────────────────────┐
                         │      MuJoCo Menagerie Unitree A1 Position Actuators     │
                         │      12-DoF PD Control (500 Hz Physics, 50 Hz Control)  │
                         └─────────────────────────────────────────────────────────┘
```

1. **Analytical Reference Generator:** A parametric kinematic engine computes continuous, dynamically feasible joint angles for diagonal trotting, 4-beat crawling, and high-speed bounding.
2. **Deep Residual RL Policy (PPO):** The neural network outputs *corrective residuals* ($\Delta q$) on top of the reference gait. When unperturbed, residuals remain small; when pushed or operating on 3 legs, the network dynamically widens the stance, alters foot placement, and regulates trunk roll/pitch.
3. **Rapid Motor Adaptation (RMA):** A 10-step history buffer encodes temporal joint deflections, allowing the policy to infer hidden environmental parameters (friction $\mu$, external forces $F_{\text{ext}}$, payload mass) online.

---

## 📐 Deep Reinforcement Learning & Mathematical Formulation

### 1. PPO Clipped Surrogate Objective & GAE
We optimize the stochastic policy $\pi_\theta(a_t | s_t)$ and value function $V_\phi(s_t)$ using Proximal Policy Optimization (PPO). The policy loss maximizes the pessimistic clipped surrogate objective:

$$\mathcal{L}^{\text{CLIP}}(\theta) = \hat{\mathbb{E}}_t \left[ \min\left(r_t(\theta)\hat{A}_t, \; \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\right) \right]$$

where the probability ratio is defined as:

$$r_t(\theta) = \frac{\pi_\theta(a_t | s_t)}{\pi_{\theta_{\text{old}}}(a_t | s_t)}$$

and the clipping threshold is set to $\epsilon = 0.2$. Advantages $\hat{A}_t$ are estimated via Generalized Advantage Estimation ($\text{GAE}(\gamma, \lambda)$) with discount factor $\gamma = 0.99$ and smoothing factor $\lambda = 0.95$:

$$\hat{A}_t^{\text{GAE}} = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}^V, \quad \delta_t^V = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)$$

The joint objective combines policy surrogate loss, value function regression, and an entropy regularization bonus:

$$\mathcal{L}_{\text{total}}(\theta, \phi) = -\mathcal{L}^{\text{CLIP}}(\theta) + c_1 \hat{\mathbb{E}}_t \left[ (V_\phi(s_t) - V_t^{\text{targ}})^2 \right] - c_2 \mathcal{S}[\pi_\theta](s_t)$$

with coefficients $c_1 = 0.5$ and $c_2 = 0.005$.

---

### 2. Analytical Kinematic Reference Trajectories
For leg $i \in \{0, 1, 2, 3\}$ ($\text{FR}, \text{FL}, \text{RR}, \text{RL}$), the normalized phase angle $\theta_i(t)$ advances according to gait frequency $f$ and leg phase offset $\phi_i$:

$$\theta_i(t) = \left( \phi_{\text{gait}}(t) + \phi_i \right) \pmod{2\pi}, \quad \phi_{\text{gait}}(t + \Delta t) = \phi_{\text{gait}}(t) + 2\pi f \Delta t$$

The target joint trajectory is computed analytically:

$$\Delta q_{\text{thigh}, i} = (A_{\text{swing}} + \Delta_{\text{yaw}, i}) \cos(\theta_i) + A_{\text{lift}} \max(0, \sin(\theta_i)) + \delta_{\text{limp, pitch}}$$

$$\Delta q_{\text{calf}, i} = -2 A_{\text{lift}} \max(0, \sin(\theta_i)) + 0.05 \min(0, \sin(\theta_i))$$

$$\Delta q_{\text{hip}, i} = (A_{\text{lat}} + \Delta_{\text{hip\_turn}, i}) \cos(\theta_i) + \delta_{\text{limp, hip}}$$

* **Thigh Swing Amplitude:** $A_{\text{swing}} = 0.26 \cdot \text{clip}(|v_x| / 0.8, 0.25, 1.45) \cdot \text{sgn}(v_x)$ rad
* **Foot Vertical Clearance:** $A_{\text{lift}} = 0.24 \cdot \text{clip}(\text{activity} + 0.3, 0.7, 1.15)$ rad
* **Differential Yaw Stride:** $\Delta_{\text{yaw}, i} = \pm 0.15 \cdot \omega_{z,\text{cmd}}$ (opposing left vs. right legs for sharp pivoting)

---

### 3. Residual Action Space & Smoothing Filter
To prevent high-frequency joint chattering and motor wear, raw neural network outputs $a_t \in [-1, 1]^{12}$ pass through an exponential low-pass filter:

$$\tilde{a}_t = \beta a_t + (1 - \beta) \tilde{a}_{t-1}, \quad \beta = 0.65$$

The final actuator position setpoint tracked by MuJoCo's PD position actuators ($K_p = 100$) is:

$$q_{\text{target}} = q_{\text{ref}}(t) + \alpha \odot \tilde{a}_t$$

where $\alpha = [0.20, 0.28, 0.28]\text{ rad}$ per leg for $[\text{hip}, \text{thigh}, \text{calf}]$.

---

### 4. Closed-Loop 360° Waypoint Steering Dynamics
The robot calculates the vector from its center of mass $(x, y)$ to the active target grid cell $(x_{\text{bait}}, y_{\text{bait}})$:

$$d = \sqrt{(x_{\text{bait}} - x)^2 + (y_{\text{bait}} - y)^2}, \quad \psi_{\text{target}} = \text{atan2}(y_{\text{bait}} - y, \; x_{\text{bait}} - x)$$

The shortest wrapped heading error $\Delta \psi \in [-\pi, \pi]$ relative to the robot's trunk yaw $\psi_{\text{robot}}$ is:

$$\Delta \psi = (\psi_{\text{target}} - \psi_{\text{robot}} + \pi) \pmod{2\pi} - \pi$$

**Adaptive Speed and Yaw Steering Law:**

$$\omega_{z,\text{cmd}} = \text{clip}(1.8 \cdot \Delta \psi, \; -1.2, \; 1.2)\text{ rad/s}$$

$$v_{x,\text{cmd}} = v_{\text{max}} \cdot \max(0.20, \; \cos(\Delta \psi)) \cdot \text{clip}(d / 0.5, \; 0.3, \; 1.0)$$

When the target is perpendicular or behind the robot ($|\Delta \psi| > 43^\circ$), forward velocity drops and the robot executes a zero-radius turn; once aligned, it accelerates forward into the cell.

---

### 5. 3-Legged Limp-Mode Center-of-Mass Compensation
When leg $k \in \{0, 1, 2, 3\}$ suffers an actuator failure ($h_k = 0$):

1. **Joint Retraction:** Target angles for leg $k$ are set to $q_{\text{thigh}} = 1.35\text{ rad}$, $q_{\text{calf}} = -2.45\text{ rad}$ (tucked up away from the terrain), and policy residuals on leg $k$ are clamped to zero: $\tilde{a}_{t, [3k:3k+3]} = 0$.
2. **Tripod Center-of-Mass Shift:** The surviving partner leg on the same lateral side applies an inward hip bias $\delta_{\text{limp, hip}} = \pm 0.10\text{ rad}$ to shift the support polygon directly underneath the robot's centerline.
3. **Tripod Cadence Boost:** Stepping frequency is increased from $2.2\text{ Hz} \to 2.5\text{ Hz}$ to maintain dynamic hopping momentum.

---

### 6. RMA 10-Step Proprioceptive History Window
A circular buffer stores joint tracking errors $\Delta q_t = q_t - q_{\text{ref},t}$ and velocities $\dot{q}_t$:

$$\mathcal{H}_t = \{ (\Delta q_{t-k}, \; \dot{q}_{t-k}) \}_{k=0}^{H-1}, \quad H = 10 \quad (0.20\text{ seconds at } 50\text{ Hz})$$

Linear-weighted temporal features are extracted:

$$z_{\text{err}} = \frac{1}{H} \sum_{k=0}^{H-1} w_k \Delta q_{t-k}, \quad z_{\text{vel}} = \frac{1}{H} \sum_{k=0}^{H-1} w_k \dot{q}_{t-k}, \quad w_k = 0.5 + \frac{k}{H-1}$$

This 24-dimensional vector provides rate-of-deflection and foot-slip context to the policy network, allowing it to differentiate between low-friction ground slip ($\mu \approx 0.45$) and external trunk force impulses ($F_{\text{ext}} > 60\text{ N}$).

---

### 7. Multi-Objective Reward Formulation
The scalar reward function at step $t$ balances goal tracking, biomechanical regularities, energy efficiency, and survival:

$$r_t = 3.0 r_{\text{vel}} + r_{\text{prog}} + 2.0 r_{\text{gait}} + 1.5 r_{\text{yaw}} + r_{\text{lat}} + r_{\text{orient}} + r_{\text{height}} + r_{\text{smooth}} + r_{\text{torque}} + r_{\text{alive}}$$

| Reward Term | Mathematical Formulation | Objective & Physical Rationale |
| :--- | :--- | :--- |
| **Velocity Tracking ($r_{\text{vel}}$)** | $\exp\left(-\frac{(v_{x,\text{local}} - v_{x,\text{cmd}})^2}{0.22}\right)$ | Tracks commanded forward speed in the robot's local frame. |
| **Bait Cell Progress ($r_{\text{prog}}$)** | $2.5 v_{x} + 18.0 (d_{t-1} - d_t) + R_{\text{bonus}}$ | Rewards decreasing distance to bait cell ($R_{\text{bonus}} = +25.0$, $+35.0$ in limp mode). |
| **Gait Regularity ($r_{\text{gait}}$)** | $\exp\left(-\frac{\|q - q_{\text{ref}}\|_2^2}{0.50}\right)$ | Preserves cyclic phase coordination across legs. |
| **Yaw Alignment ($r_{\text{yaw}}$)** | $\exp\left(-\frac{(\omega_{z,\text{local}} - \omega_{z,\text{cmd}})^2}{0.30}\right)$ | Accurately tracks sharp heading commands. |
| **Lateral & Vertical Damping ($r_{\text{lat}}$)** | $-1.5 (v_{y} - v_{y,\text{cmd}})^2 - 0.7 v_{z}^2$ | Penalizes uncontrolled sideways drift and vertical bouncing. |
| **Trunk Posture ($r_{\text{orient}}$)** | $-w_{\text{orient}} \|g_{\text{proj}, xy}\|_2^2$ | Enforces level torso roll and pitch ($w = 2.5$ normal, $1.8$ limp). |
| **Torso Height ($r_{\text{height}}$)** | $-8.0 (\max(0, |z_{\text{trunk}} - 0.28| - 0.03))^2$ | Holds nominal trunk clearance around $0.28\text{ m}$. |
| **Action Smoothness ($r_{\text{smooth}}$)** | $-0.01 \|a_t - a_{t-1}\|_2^2 - 0.01 \|a_t\|_2^2$ | Minimizes high-frequency joint jitter and actuator strain. |
| **Energy Regularization ($r_{\text{torque}}$)** | $-3 \times 10^{-5} \|\tau\|_2^2$ | Penalizes unnecessary motor torque expenditure. |
| **Survival Bonus ($r_{\text{alive}}$)** | $+0.20$ (4-legged) / $+0.35$ (3-legged) | Rewards staying upright ($z > 0.15\text{ m}$, tilt $< 65^\circ$). |

---

## 🧩 Unified Technical Architecture & State Spaces

### Observation Space (78 Dimensions)
The observation vector preserves full backward compatibility with earlier 47-dimensional checkpoints:

```
[0:3]   Base Angular Velocity in local frame (ω_x, ω_y, ω_z)
[3:6]   Projected Gravity Vector in body frame (R_world_to_body * [0, 0, -1]^T)
[6:9]   Target Velocity Command (v_x, v_y, ω_z)
[9:11]  Cyclic Gait Clock (sin(φ_gait), cos(φ_gait))
[11:23] Joint Position Tracking Errors (q_actual - q_ref) [12 DoF]
[23:35] Joint Velocities scaled by 0.1 (dq) [12 DoF]
[35:47] Previous Executed Action a_{t-1} [12 DoF]
─────────────────────────────────────────────────────────────────────────── (47 Core Dims)
[47:50] Gait Mode One-Hot Vector ([is_walk, is_trot, is_bound])
[50:54] Leg Health Status Mask ([FR, FL, RR, RL], 1.0=ok, 0.0=fault)
[54:66] RMA 10-Step Exponentially Weighted Joint Error Encoding [12 DoF]
[66:78] RMA 10-Step Exponentially Weighted Joint Velocity Encoding [12 DoF]
─────────────────────────────────────────────────────────────────────────── (Total: 78 Dims)
```

---

## 📈 Experimental Results & Benchmark Data

### PPO Training Convergence Metrics

The unified model was trained on the Ada HPC cluster across 8 vectorized simulation environments ($1,507,328$ total steps in $5,390$ seconds at $\approx 279\text{ FPS}$):

```text
=============================================================================
EVALUATION METRICS AT 1,500,000 TIMESTEPS
=============================================================================
Mean Episode Length:     1,000.00 +/- 0.00 steps (100% full survival)
Mean Evaluation Return:  5,770.00
Explained Variance:      0.914
Value Function Loss:     627.0
Policy Gradient Loss:    -0.0184
Approximate KL:          0.0190
Clip Fraction:           0.241
Checkpoints Saved:       ./checkpoints/final_unified_model.zip
=============================================================================
```

---

### Force Sweep Benchmark: 4-Legged vs. 3-Legged Limp Mode

We benchmarked policy robustness under randomized external 3D force impulses applied directly to the trunk body across increasing force levels ($0\text{ N} \to 120\text{ N}$), evaluating both **Standard 4-Legged Multi-Gait Mode** and **3-Legged Actuator Failure Limp Mode**:

| External Force Impulse | 4-Leg Survival (%) | 4-Leg Avg Baits / Ep | 4-Leg Return | 3-Leg Limp Survival (%) | 3-Leg Limp Avg Baits / Ep | 3-Leg Limp Return |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0 N** | **100.0%** | **5.0** | `6057.5` | **100.0%** | **2.4** | `4685.2` |
| **20 N** | **100.0%** | **4.6** | `5927.0` | **80.0%** | **1.4** | `3664.7` |
| **40 N** | **100.0%** | **4.4** | `5934.2` | **80.0%** | **3.0** | `4184.3` |
| **60 N** | **100.0%** | **4.6** | `5880.0` | **60.0%** | **1.2** | `2888.4` |
| **80 N** | **100.0%** | **4.2** | `5552.1` | **60.0%** | **1.6** | `3708.3` |
| **100 N** | **60.0%** | **3.4** | `3984.7` | **80.0%** | **2.4** | `4338.1` |
| **120 N** | **80.0%** | **3.8** | `4547.0` | **20.0%** | **1.2** | `2088.6` |

#### Key Empirical Findings:
* **Nominal Multi-Gait Agility (0 N):** The robot catches an average of **5.0 baits per 20 seconds** (1 bait every 4 seconds) with zero falls, dynamically shifting between Walk, Trot, and Bound.
* **Push Rejection (20 N – 80 N):** 4-legged survival remains **100.0%**, maintaining throughput of $4.2 - 4.6$ baits per episode even under 80 N impacts (~70% of the robot's body weight).
* **Fault-Tolerant 3-Legged Adaptation:** When an entire leg is severed, the robot achieves **100.0% survival at 0 N** and collects **2.4 baits on 3 legs**. It retains **60%–80% survival** against pushes up to 100 N.
* **Hardware Torque Boundary (120 N):** At 120 N (~10 kgf impact), the surviving 3 leg motors reach their physical torque limit ($33.5\text{ Nm}$), establishing the platform's physical operational ceiling.

---

### Ablation: Baseline Open-Loop Trot vs. Closed-Loop RL

| Evaluated Metric | Open-Loop Kinematic Trot (`action = 0`) | Closed-Loop Unified RL Policy |
| :--- | :--- | :--- |
| **Control Architecture** | Static sinusoidal clock trajectory | 50 Hz closed-loop feedback ($78 \to 12$) |
| **Autonomous Target Tracking** | Blind forward path; cannot steer | Dynamic 360° waypoint steering to grid cells |
| **Gait Modality** | Fixed diagonal trot only | Automatic switching: Walk, Trot, Bound |
| **Lateral Push Limit** | Tips over at $>30\text{ N}$ | Survives pushes up to $100\text{ N} - 120\text{ N}$ |
| **Actuator Failure Response** | Instant collapse when 1 leg is disabled | Hops and collects baits on 3 legs |
| **Slippery Floor Adaptation** | Foot slips in place on $\mu < 0.6$ | RMA history infers slip and widens stance |

---

## 📁 Repository Structure

```text
.
├── setup_robot.py              # Downloads Unitree A1 model from MuJoCo Menagerie
├── quadruped_env.py            # Unified Gymnasium Env (Multi-Gait, 3-Leg Limp, RMA, Bait Nav)
├── train.py                    # Vectorized PPO pipeline with warm-start weight surgery & curriculum
├── evaluate.py                 # Benchmarking script (4-legged vs. 3-legged limp force sweeps)
├── record_video.py             # Headless EGL video recorder with live telemetry HUD overlay
├── record_progression.py       # Full 2+ minute (126s) 7-milestone chronological progression video generator
├── view_interactive.py         # Real-time 3D visualizer with mouse camera control & event logging
├── run_train.sh                # Portable launcher bash script for training
├── checkpoints/
│   ├── best_model/best_model.zip       # Best evaluation model
│   ├── final_model.zip                 # 3M-step Phase 2 model
│   └── final_unified_model.zip         # 78-dim Phase 3 unified research model
├── quadruped_unified_demo.mp4  # Final showcase demo (Multi-Gait + 3-Leg Limp Mode + Bait)
├── training_progression_2min.mp4 # Full 2+ minute (126s) Step 0-to-Final chronological progression
├── PROJECT_GUIDE.md            # Technical log & diagnostic notes
└── README.md                   # Complete documentation file
```

---

## 💻 Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/saninduhansara/Quadruped-Gait-Adaptation-under-External-Perturbations.git
cd Quadruped-Gait-Adaptation-under-External-Perturbations
```

### 2. Set Up Python Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\Activate.ps1
```

### 3. Install Dependencies
```bash
# PyTorch (install CUDA build matching your GPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# MuJoCo, Gymnasium, Stable-Baselines3, and rendering tools
pip install mujoco gymnasium stable-baselines3 tensorboard tqdm rich imageio imageio-ffmpeg
```

### 4. Download Robot Model
```bash
python3 setup_robot.py
```

---

## 🚀 Step-by-Step Execution Guide

### 1. Launching Interactive 3D Real-Time Simulator
Run the live interactive 3D visualizer locally with real-time mouse camera controls:

```bash
python3 view_interactive.py --model-path "./checkpoints/final_unified_model.zip" --demo-limp
```
* **Mouse Controls:** Left-click + drag to rotate camera 360°, Right-click to pan, Scroll wheel to zoom.
* **Console Notifications:** Prints real-time events for every `[GAIT SWITCH]`, `[3-LEG LIMP MODE ACTIVATED]`, and `[BAIT COLLECTED]`.

---

### 2. Training & Fine-Tuning the Unified Model
Launch multi-environment PPO training across 8 parallel simulation workers with adaptive curriculum:

```bash
# Run via portable shell script:
chmod +x run_train.sh
./run_train.sh

# Or directly via Python:
python3 train.py --num-envs 8 --push-force 60.0 --fault-prob 0.25 --total-timesteps 1500000 --resume-from ./checkpoints/best_model/best_model.zip --device auto
```

---

### 3. Running Benchmarks (4-Leg & 3-Leg Limp Mode)
Evaluate policy survival rate, bait collection throughput, and return across increasing external push forces ($0\text{ N} \to 120\text{ N}$):

```bash
# 1. Benchmark 4-Legged Multi-Gait Mode:
python3 evaluate.py --model-path ./checkpoints/final_unified_model.zip

# 2. Benchmark 3-Legged Actuator Failure Limp Mode:
python3 evaluate.py --model-path ./checkpoints/final_unified_model.zip --test-limp
```

---

### 4. Generating 2+ Minute Showcase Videos

#### Option A: 18-Second Showcase with Live Telemetry HUD
Demonstrates Walk, Trot, and Bound transitions in the first half, injects an actuator failure at step 450, and showcases 3-legged limp mode:
```bash
export MUJOCO_GL="egl"  # On headless servers
python3 record_video.py --model-path ./checkpoints/final_unified_model.zip --output quadruped_unified_demo.mp4 --steps 900 --limp-step 450
```

#### Option B: Full 2+ Minute (126s) Chronological Progression Video
Automatically stitches together 7 distinct milestones from Step 0 to final with a real-time elapsed timer (`MM:SS / 02:06`):
```bash
export MUJOCO_GL="egl"  # On headless servers
python3 record_progression.py --checkpoint-dir ./checkpoints --output training_progression_2min.mp4 --duration 126 --fps 30
```

---

## 🛠️ Key Engineering Problems & Solutions

### 1. Multiprocessing XML Overwrite Race Condition
* **Problem:** Parallel workers spawned by `SubprocVecEnv` simultaneously attempted to write `scene_bait.xml`, causing workers to parse an empty (0-byte) file during mid-write truncation.
* **Solution:** [`_ensure_bait_scene_xml()`](./quadruped_env.py) writes to a PID-unique temp file (`scene_bait.xml.tmp.<pid>`) and executes an atomic POSIX `os.replace()`. It is also pre-generated once in the main process before worker spawning.

### 2. Headless GPU Offscreen Rendering
* **Problem:** `GLFWError: (65550) b'X11: The DISPLAY environment variable is missing'` when running on headless GPU nodes.
* **Solution:** Configure `export MUJOCO_GL="egl"` before importing MuJoCo, routing rendering calls directly to offscreen NVIDIA EGL GPU buffers.

### 3. Warm-Start Weight Surgery (47D to 78D Transfer)
* **Problem:** Expanding the observation vector from 47 to 78 dimensions to support Multi-Gait, Limp Mode, and RMA history would typically invalidate pretrained checkpoints.
* **Solution:** Positioned the 47 core dimensions identically at the front of the observation vector and implemented [`transfer_weights_47_to_78()`](./train.py), which copied pretrained weights for the first 47 inputs and initialized the new 31 research columns near zero. The policy retained full walking balance from step 0 of retraining.

### 4. Memory-Safe Streaming for Long Video Generation
* **Problem:** Rendering 2+ minute videos at 30–50 FPS requires buffering over 3,700 high-resolution frames, risking host RAM exhaustion.
* **Solution:** Streamed frames directly to disk via `imageio.get_writer()` chunk writes in [`record_progression.py`](./record_progression.py), ensuring constant low memory usage regardless of video duration.

---

## 📄 License & Citation

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

If you use this work or findings in your research or project, please consider citing:

```bibtex
@misc{quadruped_gait_adaptation_2026,
  author = {Sanindu Hansara},
  title = {Quadruped Gait Adaptation under External Perturbations & Actuator Failure Recovery},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/saninduhansara/Quadruped-Gait-Adaptation-under-External-Perturbations}}
}
```
