#!/bin/bash
# Shell script to fine-tune quadruped 360-degree sharp-turn bait navigation on Ada server

# 1. Activate Python virtual environment
source /tmp/quad_rl_new/bin/activate

# 2. Select GPU (0, 1, or 2)
export CUDA_VISIBLE_DEVICES=1

# 3. Headless rendering
export MUJOCO_GL="egl"

# 4. Navigate to project directory
cd /new-home/e22/e22130/projects/quad

# 5. Fine-tune from existing best_model.zip for 1,000,000 steps in 360-degree bait mode
python3 train.py --num-envs 8 --push-force 60.0 --total-timesteps 1000000 --resume-from ./checkpoints/best_model/best_model.zip --device auto
