import os
import sys

# Crucial: Set headless rendering backend before importing mujoco
if "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "egl"

import argparse
import imageio
import numpy as np
import mujoco

from stable_baselines3 import PPO

from quadruped_env import QuadrupedPerturbationEnv

def record_rollout(model_path, output_video="quadruped_adaptation.mp4", max_steps=500):
    print(f"[INFO] Loading model from: {model_path}")
    model = PPO.load(model_path)

    # Initialize environment
    env = QuadrupedPerturbationEnv(
        perturbation_prob=0.03,  # Push every ~30 steps
        max_push_force=150.0,
        push_duration_steps=8
    )

    # MuJoCo Offscreen Renderer
    width, height = 640, 480
    renderer = mujoco.Renderer(env.model, height=height, width=width)
    
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    camera.trackbodyid = env.trunk_body_id
    camera.distance = 2.0
    camera.elevation = -15
    camera.azimuth = 90

    frames = []
    obs, _ = env.reset(seed=42)

    print(f"[INFO] Recording {max_steps} steps...")
    for step in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        # Update tracking camera position
        renderer.update_scene(env.data, camera=camera)
        pixels = renderer.render()
        frames.append(pixels)

        if terminated:
            print(f"[INFO] Robot fell at step {step}, resetting...")
            obs, _ = env.reset()

    # Save to MP4
    print(f"[INFO] Saving video to {output_video}...")
    imageio.mimsave(output_video, frames, fps=50)
    print(f"[SUCCESS] Video saved: {output_video} ({len(frames)} frames)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="./checkpoints/best_model/best_model.zip")
    parser.add_argument("--output", type=str, default="quadruped_adaptation.mp4")
    parser.add_argument("--steps", type=int, default=500)
    args = parser.parse_args()

    record_rollout(args.model_path, args.output, args.steps)
