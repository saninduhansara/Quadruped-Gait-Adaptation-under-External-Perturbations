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


def record_rollout(
    model_path,
    output_video="quadruped_bait_navigation.mp4",
    max_steps=800,
    push_force=50.0,
    use_bait=True,
):
    model = None
    if model_path and os.path.exists(model_path):
        print(f"[INFO] Loading trained PPO model from: {model_path}")
        model = PPO.load(model_path)
    else:
        print(f"[WARN] Model not found at '{model_path}'. Running kinematic reference gait (zero residual).")

    # Initialize environment with 360-degree random cell bait navigation
    env = QuadrupedPerturbationEnv(
        perturbation_prob=0.02,
        max_push_force=push_force,
        push_duration_steps=6,
        use_bait=use_bait,
    )

    # MuJoCo Offscreen Renderer
    width, height = 640, 480
    renderer = mujoco.Renderer(env.model, height=height, width=width)

    # Wide-angle elevated tracking camera so both the robot and random bait cells are clearly visible
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    camera.trackbodyid = env.trunk_body_id
    camera.distance = 3.6
    camera.elevation = -32
    camera.azimuth = 115

    frames = []
    obs, _ = env.reset(seed=42)
    if not use_bait:
        env.command = np.array([0.85, 0.0, 0.0], dtype=np.float32)
        obs = env._get_obs()
    else:
        print(f"[BAIT SPAWNED] Target cell #1 at (x={env.bait_pos[0]:.1f} m, y={env.bait_pos[1]:.1f} m)")

    print(f"[INFO] Recording {max_steps} steps ({max_steps * env.control_dt:.1f}s)...")
    falls = 0
    total_baits = 0
    prev_baits = 0

    for step in range(max_steps):
        if model is not None:
            action, _ = model.predict(obs, deterministic=True)
        else:
            action = np.zeros(env.num_joints, dtype=np.float32)

        obs, reward, terminated, truncated, info = env.step(action)

        if use_bait and env.baits_collected > prev_baits:
            total_baits += 1
            prev_baits = env.baits_collected
            print(
                f"  [BAIT COLLECTED #{total_baits}] at step {step + 1}! "
                f"New bait spawned at cell (x={env.bait_pos[0]:.1f} m, y={env.bait_pos[1]:.1f} m)"
            )

        # Render frame
        renderer.update_scene(env.data, camera=camera)
        pixels = renderer.render()
        frames.append(pixels)

        if (step + 1) % 100 == 0:
            rx, ry = float(env.data.qpos[0]), float(env.data.qpos[1])
            bx, by = float(env.bait_pos[0]), float(env.bait_pos[1])
            dist = np.hypot(bx - rx, by - ry)
            print(
                f"  Step {step + 1:>4d}/{max_steps} | Robot=(x={rx:>5.2f}, y={ry:>5.2f}) | "
                f"Bait Cell=(x={bx:>4.1f}, y={by:>4.1f}) | Dist={dist:>4.2f}m | Baits={total_baits}"
            )

        if terminated:
            falls += 1
            print(f"[INFO] Robot fell at step {step + 1}, resetting...")
            obs, _ = env.reset()
            prev_baits = 0
            if not use_bait:
                env.command = np.array([0.85, 0.0, 0.0], dtype=np.float32)
                obs = env._get_obs()

    # Save to MP4
    print(f"[INFO] Total Baits Collected: {total_baits} | Total Falls: {falls}")
    print(f"[INFO] Saving video to {output_video}...")
    imageio.mimsave(output_video, frames, fps=50)
    print(f"[SUCCESS] Video saved: {output_video} ({len(frames)} frames)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="./checkpoints/best_model/best_model.zip")
    parser.add_argument("--output", type=str, default="quadruped_bait_navigation.mp4")
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--push-force", type=float, default=50.0)
    parser.add_argument("--no-bait", action="store_true", help="Disable random bait navigation")
    args = parser.parse_args()

    record_rollout(args.model_path, args.output, args.steps, args.push_force, use_bait=not args.no_bait)
