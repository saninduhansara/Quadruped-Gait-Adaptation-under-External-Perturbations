#!/bin/bash
# Shell script to train the Unified Quadruped RL System on Ada server:
# (360° Bait Navigation + 3-in-1 Multi-Gait + 3-Legged Fault Recovery + 10-Step RMA History)

# 1. Activate Python virtual environment
source /tmp/quad_rl_new/bin/activate

# 2. Select GPU (0, 1, or 2)
export CUDA_VISIBLE_DEVICES=1

# 3. Headless rendering
export MUJOCO_GL="egl"

# 4. Navigate to project directory
cd /new-home/e22/e22130/projects/quad

# 5. Warm-start from existing best_model.zip into the 78-dim unified policy and train for 1,500,000 steps
python3 train.py --num-envs 8 --push-force 60.0 --fault-prob 0.25 --total-timesteps 1500000 --resume-from ./checkpoints/best_model/best_model.zip --device auto
