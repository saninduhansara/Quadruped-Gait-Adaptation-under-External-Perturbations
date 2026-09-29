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

def record_rollout(model_path, output_video="quadruped_adaptation_v3.mp4", max_steps=600, push_force=55.0):
    model = None
    if model_path and os.path.exists(model_path):
        print(f"[INFO] Loading trained PPO model from: {model_path}")
        model = PPO.load(model_path)
    else:
        print(f"[WARN] Model not found at '{model_path}'. Running kinematic reference trotting gait (zero residual).")

    # Initialize environment
    env = QuadrupedPerturbationEnv(
        perturbation_prob=0.02,  # Periodic external push perturbations
        max_push_force=push_force,
        push_duration_steps=6
    )

    # MuJoCo Offscreen Renderer
    width, height = 640, 480
    renderer = mujoco.Renderer(env.model, height=height, width=width)
    
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    camera.trackbodyid = env.trunk_body_id
    camera.distance = 2.2
    camera.elevation = -18
    camera.azimuth = 90

    frames = []
    obs, _ = env.reset(seed=42)
    # Lock straight-forward walking command so robot walks cleanly cell-by-cell
    env.command = np.array([0.85, 0.0, 0.0], dtype=np.float32)
    obs = env._get_obs()

    print(f"[INFO] Commanded forward velocity (vx): {env.command[0]:.2f} m/s")
    print(f"[INFO] Recording {max_steps} steps ({max_steps * env.control_dt:.1f}s)...")
    falls = 0
    max_x = 0.0

    for step in range(max_steps):
        if model is not None:
            action, _ = model.predict(obs, deterministic=True)
        else:
            action = np.zeros(env.num_joints, dtype=np.float32)

        obs, reward, terminated, truncated, info = env.step(action)
        max_x = max(max_x, float(env.data.qpos[0]))

        # Update tracking camera position
        renderer.update_scene(env.data, camera=camera)
        pixels = renderer.render()
        frames.append(pixels)

        if (step + 1) % 100 == 0:
            print(f"  Step {step + 1:>4d}/{max_steps} | Position x = {env.data.qpos[0]:>5.2f} m | Vel vx = {env.data.qvel[0]:>4.2f} m/s")

        if terminated:
            falls += 1
            print(f"[INFO] Robot fell at step {step} (x = {env.data.qpos[0]:.2f} m), resetting...")
            obs, _ = env.reset()
            env.command = np.array([0.85, 0.0, 0.0], dtype=np.float32)
            obs = env._get_obs()

    # Save to MP4
    print(f"[INFO] Peak forward distance reached: x = {max_x:.2f} m | Total falls: {falls}")
    print(f"[INFO] Saving video to {output_video}...")
    imageio.mimsave(output_video, frames, fps=50)
    print(f"[SUCCESS] Video saved: {output_video} ({len(frames)} frames)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="./checkpoints/best_model/best_model.zip")
    parser.add_argument("--output", type=str, default="quadruped_adaptation_v3.mp4")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--push-force", type=float, default=55.0)
    args = parser.parse_args()

    record_rollout(args.model_path, args.output, args.steps, args.push_force)

