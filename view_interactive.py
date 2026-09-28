"""
Interactive 3D Real-time Visualizer for Quadruped Gait Adaptation.
Run this on your local Windows PC to watch the robot walk in real-time with an interactive 3D camera.
"""
import os
import time
import argparse
import mujoco
import mujoco.viewer
import numpy as np
from stable_baselines3 import PPO

from quadruped_env import QuadrupedPerturbationEnv

def run_interactive_viewer(model_path):
    print(f"[INFO] Loading model from: {model_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at: {model_path}")

    model = PPO.load(model_path)
    env = QuadrupedPerturbationEnv(
        perturbation_prob=0.03,
        max_push_force=150.0,
        push_duration_steps=8
    )

    obs, _ = env.reset()

    print("[INFO] Launching MuJoCo 3D Interactive Viewer...")
    print("Controls: Left click + drag to rotate, Right click + drag to pan, Scroll to zoom.")

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        while viewer.is_running():
            step_start = time.time()

            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            # Sync simulation state to 3D GUI window
            viewer.sync()

            if terminated or truncated:
                obs, _ = env.reset()

            # Maintain real-time 50 Hz speed
            time_until_next_step = env.control_dt - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="./checkpoints/best_model/best_model.zip")
    args = parser.parse_args()

    run_interactive_viewer(args.model_path)
