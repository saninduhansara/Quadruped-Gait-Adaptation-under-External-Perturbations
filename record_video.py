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
from record_progression import draw_text


def annotate_telemetry_hud(
    frame: np.ndarray,
    gait_name: str,
    disabled_leg: str,
    baits_collected: int,
    bait_pos: np.ndarray,
    speed: float,
    is_push_active: bool,
) -> np.ndarray:
    """Draw a sleek live telemetry HUD overlay onto the rendered video frame."""
    out = frame.copy()
    h, w, _ = out.shape

    banner_h = 56
    out[:banner_h, :] = (out[:banner_h, :].astype(np.float32) * 0.25).astype(np.uint8)

    # Color-coded top bar based on gait/limp state
    if disabled_leg != "NONE":
        out[:4, :] = (230, 50, 210)  # Magenta bar for 3-Legged Fault-Tolerant Limp Mode
        mode_text = f"3-LEG LIMP MODE ({disabled_leg} DISABLED)"
        mode_color = (255, 130, 245)
    elif gait_name == "BOUND":
        out[:4, :] = (255, 165, 40)  # Orange bar for High-Speed Bound
        mode_text = "MULTI-GAIT: BOUND (HIGH SPEED)"
        mode_color = (255, 200, 90)
    elif gait_name == "WALK":
        out[:4, :] = (70, 190, 255)  # Cyan bar for 4-Beat Walk / Sharp Turn
        mode_text = "MULTI-GAIT: 4-BEAT WALK (TURN/ALIGN)"
        mode_color = (120, 220, 255)
    else:
        out[:4, :] = (50, 225, 120)  # Green bar for Diagonal Trot
        mode_text = "MULTI-GAIT: DIAGONAL TROT"
        mode_color = (100, 245, 150)

    draw_text(out, mode_text, x=14, y=11, scale=2, color=mode_color)
    sub = f"BAITS: {baits_collected} | TARGET CELL: ({bait_pos[0]:.0f},{bait_pos[1]:.0f}) | SPD: {speed:.2f} M/S"
    draw_text(out, sub, x=14, y=33, scale=2, color=(235, 245, 255))

    if is_push_active:
        badge_w = 185
        out[12:44, w - badge_w - 12 : w - 12] = (215, 35, 35)
        draw_text(out, "EXTERNAL PUSH!", x=w - badge_w - 2, y=21, scale=2, color=(255, 255, 255))

    return out


def predict_compatible(model: PPO, obs: np.ndarray):
    """Support both 47-dim legacy checkpoints and 78-dim unified checkpoints seamlessly."""
    expected_dim = model.observation_space.shape[0]
    return model.predict(obs[:expected_dim], deterministic=True)


def record_rollout(
    model_path,
    output_video="quadruped_unified_demo.mp4",
    max_steps=900,
    push_force=55.0,
    use_bait=True,
    limp_demo_step=450,
):
    model = None
    if model_path and os.path.exists(model_path):
        print(f"[INFO] Loading trained PPO model from: {model_path}")
        model = PPO.load(model_path)
    else:
        print(f"[WARN] Model not found at '{model_path}'. Running reference gait (zero residual).")

    env = QuadrupedPerturbationEnv(
        perturbation_prob=0.02,
        max_push_force=push_force,
        push_duration_steps=6,
        use_bait=use_bait,
        leg_failure_prob=0.0,  # Controlled deterministically by limp_demo_step for clear video demo
    )

    width, height = 640, 480
    renderer = mujoco.Renderer(env.model, height=height, width=width)

    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    camera.trackbodyid = env.trunk_body_id
    camera.distance = 3.5
    camera.elevation = -30
    camera.azimuth = 115

    frames = []
    obs, _ = env.reset(seed=42)
    print(f"[BAIT SPAWNED] Target cell #1 at (x={env.bait_pos[0]:.1f} m, y={env.bait_pos[1]:.1f} m)")
    print(f"[INFO] Recording {max_steps} steps ({max_steps * env.control_dt:.1f}s)...")

    falls = 0
    total_baits = 0
    prev_baits = 0
    last_gait = None

    for step in range(max_steps):
        # Trigger 3-legged actuator failure halfway through the video to showcase limp adaptation!
        if limp_demo_step > 0 and step == limp_demo_step:
            env.set_disabled_leg(0)  # Break Front-Right (FR) leg
            print(f"\n  [ACTUATOR FAILURE INJECTED at step {step}] -> Front-Right (FR) Leg Disabled! Switching to 3-Legged Limp Mode...\n")

        if model is not None:
            action, _ = predict_compatible(model, obs)
        else:
            action = np.zeros(env.num_joints, dtype=np.float32)

        obs, reward, terminated, truncated, info = env.step(action)

        gait_name = info.get("gait_mode", "TROT")
        disabled_leg = info.get("disabled_leg", "NONE")

        if gait_name != last_gait:
            last_gait = gait_name

        if use_bait and env.baits_collected > prev_baits:
            total_baits += 1
            prev_baits = env.baits_collected
            print(
                f"  [BAIT COLLECTED #{total_baits}] at step {step + 1} (Mode: {gait_name}, Broken Leg: {disabled_leg})! "
                f"Next bait -> cell (x={env.bait_pos[0]:.1f} m, y={env.bait_pos[1]:.1f} m)"
            )

        renderer.update_scene(env.data, camera=camera)
        raw_pixels = renderer.render()

        speed = float(np.hypot(env.data.qvel[0], env.data.qvel[1]))
        annotated = annotate_telemetry_hud(
            raw_pixels,
            gait_name=gait_name,
            disabled_leg=disabled_leg,
            baits_collected=total_baits,
            bait_pos=env.bait_pos,
            speed=speed,
            is_push_active=bool(info.get("is_push_active", False)),
        )
        frames.append(annotated)

        if (step + 1) % 100 == 0:
            rx, ry = float(env.data.qpos[0]), float(env.data.qpos[1])
            bx, by = float(env.bait_pos[0]), float(env.bait_pos[1])
            dist = np.hypot(bx - rx, by - ry)
            print(
                f"  Step {step + 1:>4d}/{max_steps} | Gait={gait_name:<5s} | LegFault={disabled_leg:<4s} | "
                f"Robot=({rx:>5.2f},{ry:>5.2f}) | Bait=({bx:>4.1f},{by:>4.1f}) | Baits={total_baits}"
            )

        if terminated:
            falls += 1
            was_limping = env.disabled_leg_idx
            print(f"[INFO] Robot fell at step {step + 1}, resetting...")
            obs, _ = env.reset()
            if was_limping >= 0:
                env.set_disabled_leg(was_limping)
            prev_baits = 0

    renderer.close()
    print(f"[INFO] Total Baits Collected: {total_baits} | Total Falls: {falls}")
    print(f"[INFO] Saving video to {output_video}...")
    imageio.mimsave(output_video, frames, fps=50)
    print(f"[SUCCESS] Video saved: {output_video} ({len(frames)} frames)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="./checkpoints/best_model/best_model.zip")
    parser.add_argument("--output", type=str, default="quadruped_unified_demo.mp4")
    parser.add_argument("--steps", type=int, default=900)
    parser.add_argument("--push-force", type=float, default=55.0)
    parser.add_argument("--limp-step", type=int, default=450, help="Step to inject 3-leg failure (0 to disable)")
    parser.add_argument("--no-bait", action="store_true", help="Disable random bait navigation")
    args = parser.parse_args()

    record_rollout(
        args.model_path,
        args.output,
        args.steps,
        args.push_force,
        use_bait=not args.no_bait,
        limp_demo_step=args.limp_step,
    )
