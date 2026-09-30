#!/bin/bash
# Shell script to train the Unified Quadruped RL System on Ada server:
# (360° Bait Navigation + 3-in-1 Multi-Gait + 3-Legged Fault Recovery + 10-Step RMA History)

# 1. Activate Python virtual environment (adjust path to your environment)
if [ -d "/tmp/quad_rl_new" ]; then
    source /tmp/quad_rl_new/bin/activate
elif [ -d "./venv" ]; then
    source ./venv/bin/activate
fi

# 2. GPU Assignment (set to desired GPU id, e.g., 0, 1, or empty for default)
export CUDA_VISIBLE_DEVICES=0

# 3. Headless rendering backend
export MUJOCO_GL="egl"

# 4. Navigate to project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 5. Launch Unified Multi-Gait, 3-Legged Limp & RMA Training
python3 train.py --num-envs 8 --push-force 60.0 --fault-prob 0.25 --total-timesteps 1500000 --resume-from ./checkpoints/best_model/best_model.zip --device auto
