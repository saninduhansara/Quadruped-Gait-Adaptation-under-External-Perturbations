"""
Interactive 3D Real-time Visualizer for the Unified Quadruped RL System:
  - 360° Random Cell Bait Navigation
  - 3-in-1 Multi-Gait Switching (WALK -> TROT -> BOUND)
  - 3-Legged Fault-Tolerant Limp Mode
  - 10-Step RMA Proprioceptive History
"""
import os
import time
import argparse
import mujoco
import mujoco.viewer
import numpy as np
from stable_baselines3 import PPO

from quadruped_env import QuadrupedPerturbationEnv


def run_interactive_viewer(model_path, use_bait=True, demo_limp=False):
    print(f"[INFO] Loading model from: {model_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at: {model_path}")

    model = PPO.load(model_path)
    expected_dim = model.observation_space.shape[0]

    env = QuadrupedPerturbationEnv(
        perturbation_prob=0.02,
        max_push_force=55.0,
        push_duration_steps=6,
        use_bait=use_bait,
        leg_failure_prob=0.35 if demo_limp else 0.15,
    )

    obs, _ = env.reset()
    prev_baits = 0
    prev_gait = None
    prev_fault = "NONE"

    print("[INFO] Launching MuJoCo 3D Interactive Viewer...")
    print("Controls: Left click + drag to rotate, Right click + drag to pan, Scroll to zoom.")
    if use_bait:
        print(f"[BAIT SPAWNED] Target cell at (x={env.bait_pos[0]:.1f} m, y={env.bait_pos[1]:.1f} m)")

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        while viewer.is_running():
            step_start = time.time()

            action, _ = model.predict(obs[:expected_dim], deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            curr_gait = info.get("gait_mode", "TROT")
            curr_fault = info.get("disabled_leg", "NONE")

            if curr_gait != prev_gait:
                prev_gait = curr_gait
                print(f"  [GAIT SWITCH] -> {curr_gait} (vx_cmd={env.command[0]:.2f} m/s, wz_cmd={env.command[2]:.2f} rad/s)")

            if curr_fault != prev_fault:
                prev_fault = curr_fault
                if curr_fault != "NONE":
                    print(f"  [3-LEG LIMP MODE ACTIVATED] Leg '{curr_fault}' disabled! Adapting gait on 3 legs...")

            if use_bait and env.baits_collected > prev_baits:
                prev_baits = env.baits_collected
                print(
                    f"  [BAIT COLLECTED #{prev_baits}] (Gait: {curr_gait}, Broken Leg: {curr_fault})! "
                    f"Next bait -> cell (x={env.bait_pos[0]:.1f} m, y={env.bait_pos[1]:.1f} m)"
                )

            viewer.sync()

            if terminated or truncated:
                obs, _ = env.reset()
                prev_baits = 0
                prev_fault = "NONE"

            time_until_next_step = env.control_dt - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="./checkpoints/best_model/best_model.zip")
    parser.add_argument("--demo-limp", action="store_true", help="Increase frequency of 3-legged limp mode demos")
    parser.add_argument("--no-bait", action="store_true", help="Disable random bait navigation")
    args = parser.parse_args()

    run_interactive_viewer(args.model_path, use_bait=not args.no_bait, demo_limp=args.demo_limp)
