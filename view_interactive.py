"""
Interactive 3D Real-time Visualizer for Quadruped Gait Adaptation & 360° Bait Navigation.
Run this on your local Windows PC to watch the robot hunt random cell baits in real-time.
"""
import os
import time
import argparse
import mujoco
import mujoco.viewer
import numpy as np
from stable_baselines3 import PPO

from quadruped_env import QuadrupedPerturbationEnv


def run_interactive_viewer(model_path, use_bait=True):
    print(f"[INFO] Loading model from: {model_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at: {model_path}")

    model = PPO.load(model_path)
    env = QuadrupedPerturbationEnv(
        perturbation_prob=0.02,
        max_push_force=55.0,
        push_duration_steps=6,
        use_bait=use_bait,
    )

    obs, _ = env.reset()
    prev_baits = 0

    print("[INFO] Launching MuJoCo 3D Interactive Viewer...")
    print("Controls: Left click + drag to rotate, Right click + drag to pan, Scroll to zoom.")
    if use_bait:
        print(f"[BAIT SPAWNED] Target cell at (x={env.bait_pos[0]:.1f} m, y={env.bait_pos[1]:.1f} m)")

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        while viewer.is_running():
            step_start = time.time()

            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            if use_bait and env.baits_collected > prev_baits:
                prev_baits = env.baits_collected
                print(
                    f"[BAIT COLLECTED #{prev_baits}]! Next bait spawned at cell "
                    f"(x={env.bait_pos[0]:.1f} m, y={env.bait_pos[1]:.1f} m)"
                )

            # Sync simulation state to 3D GUI window
            viewer.sync()

            if terminated or truncated:
                obs, _ = env.reset()
                prev_baits = 0

            # Maintain real-time 50 Hz speed
            time_until_next_step = env.control_dt - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="./checkpoints/best_model/best_model.zip")
    parser.add_argument("--no-bait", action="store_true", help="Disable random bait navigation")
    args = parser.parse_args()

    run_interactive_viewer(args.model_path, use_bait=not args.no_bait)
