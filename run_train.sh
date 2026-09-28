#!/bin/bash
# Shell script to run quadruped training on Ada server in the background

# 1. Activate Python virtual environment
source /tmp/quad_rl_new/bin/activate

# 2. Select GPU (0, 1, or 2)
export CUDA_VISIBLE_DEVICES=1

# 3. Headless rendering
export MUJOCO_GL="egl"

# 4. Navigate to project directory
cd /new-home/e22/e22130/projects/quad

# 5. Run training
python3 train.py --num-envs 8 --push-force 150.0 --total-timesteps 5000000 --device auto
