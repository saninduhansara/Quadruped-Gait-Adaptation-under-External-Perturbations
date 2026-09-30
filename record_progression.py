"""
Generates a complete Training Progression Timelapse Video (from Step 0 to Final Trained Model)
by loading historical checkpoints saved in ./checkpoints/ in chronological order and stitching
them into a single annotated MP4 video.
"""
import os
import re
import glob
import argparse

# Crucial: Set headless rendering backend before importing mujoco
if "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "egl"

import numpy as np
import imageio
import mujoco
from stable_baselines3 import PPO

from quadruped_env import QuadrupedPerturbationEnv


# Minimal 5x7 bitmap font for pure-NumPy HUD text overlay (zero external font dependencies)
FONT_5X7 = {
    "A": ["01110", "10001", "10001", "11111", "10001", "10001", "10001"],
    "B": ["11110", "10001", "10001", "11110", "10001", "10001", "11110"],
    "C": ["01110", "10001", "10000", "10000", "10000", "10001", "01110"],
    "D": ["11100", "10010", "10001", "10001", "10001", "10010", "11100"],
    "E": ["11111", "10000", "10000", "11110", "10000", "10000", "11111"],
    "F": ["11111", "10000", "10000", "11110", "10000", "10000", "10000"],
    "G": ["01110", "10001", "10000", "10111", "10001", "10001", "01110"],
    "H": ["10001", "10001", "10001", "11111", "10001", "10001", "10001"],
    "I": ["01110", "00100", "00100", "00100", "00100", "00100", "01110"],
    "J": ["00111", "00010", "00010", "00010", "10010", "10010", "01100"],
    "K": ["10001", "10010", "10100", "11000", "10100", "10010", "10001"],
    "L": ["10000", "10000", "10000", "10000", "10000", "10000", "11111"],
    "M": ["10001", "11011", "10101", "10101", "10001", "10001", "10001"],
    "N": ["10001", "11001", "10101", "10011", "10001", "10001", "10001"],
    "O": ["01110", "10001", "10001", "10001", "10001", "10001", "01110"],
    "P": ["11110", "10001", "10001", "11110", "10000", "10000", "10000"],
    "Q": ["01110", "10001", "10001", "10001", "10101", "10010", "01101"],
    "R": ["11110", "10001", "10001", "11110", "10100", "10010", "10001"],
    "S": ["01111", "10000", "10000", "01110", "00001", "00001", "11110"],
    "T": ["11111", "00100", "00100", "00100", "00100", "00100", "00100"],
    "U": ["10001", "10001", "10001", "10001", "10001", "10001", "01110"],
    "V": ["10001", "10001", "10001", "10001", "10001", "01010", "00100"],
    "W": ["10001", "10001", "10001", "10101", "10101", "10101", "01010"],
    "X": ["10001", "10001", "01010", "00100", "01010", "10001", "10001"],
    "Y": ["10001", "10001", "01010", "00100", "00100", "00100", "00100"],
    "Z": ["11111", "00001", "00010", "00100", "01000", "10000", "11111"],
    "0": ["01110", "10001", "10011", "10101", "11001", "10001", "01110"],
    "1": ["00100", "01100", "00100", "00100", "00100", "00100", "01110"],
    "2": ["01110", "10001", "00001", "00010", "00100", "01000", "11111"],
    "3": ["11110", "00001", "00001", "01110", "00001", "00001", "11110"],
    "4": ["00010", "00110", "01010", "10010", "11111", "00010", "00010"],
    "5": ["11111", "10000", "10000", "11110", "00001", "00001", "11110"],
    "6": ["01110", "10000", "10000", "11110", "10001", "10001", "01110"],
    "7": ["11111", "00001", "00010", "00100", "01000", "01000", "01000"],
    "8": ["01110", "10001", "10001", "01110", "10001", "10001", "01110"],
    "9": ["01110", "10001", "10001", "01111", "00001", "00001", "01110"],
    ":": ["00000", "00100", "00100", "00000", "00100", "00100", "00000"],
    ".": ["00000", "00000", "00000", "00000", "00000", "00100", "00100"],
    ",": ["00000", "00000", "00000", "00000", "00100", "00100", "01000"],
    "-": ["00000", "00000", "00000", "11111", "00000", "00000", "00000"],
    "/": ["00001", "00010", "00010", "00100", "01000", "01000", "10000"],
    "(": ["00010", "00100", "01000", "01000", "01000", "00100", "00010"],
    ")": ["01000", "00100", "00010", "00010", "00010", "00100", "01000"],
    "#": ["01010", "11111", "01010", "01010", "11111", "01010", "00000"],
    "%": ["11001", "11010", "00010", "00100", "01000", "01011", "10011"],
    "+": ["00000", "00100", "00100", "11111", "00100", "00100", "00000"],
    "!": ["00100", "00100", "00100", "00100", "00100", "00000", "00100"],
    " ": ["00000", "00000", "00000", "00000", "00000", "00000", "00000"],
}


def draw_text(frame: np.ndarray, text: str, x: int, y: int, scale: int = 2, color=(255, 255, 255)):
    """Render crisp uppercase bitmap text onto an RGB frame in-place."""
    h, w, _ = frame.shape
    cursor_x = x
    for ch in text.upper():
        glyph = FONT_5X7.get(ch, FONT_5X7[" "])
        for row_idx, row_str in enumerate(glyph):
            for col_idx, bit in enumerate(row_str):
                if bit == "1":
                    y0 = y + row_idx * scale
                    y1 = min(h, y0 + scale)
                    x0 = cursor_x + col_idx * scale
                    x1 = min(w, x0 + scale)
                    if 0 <= y0 < h and 0 <= x0 < w:
                        frame[y0:y1, x0:x1] = color
        cursor_x += 6 * scale


def annotate_frame(
    frame: np.ndarray,
    stage_title: str,
    subtitle: str,
    progress_ratio: float,
    is_push_active: bool = False,
) -> np.ndarray:
    """Add a sleek top HUD banner with training stage info and push indicator."""
    out = frame.copy()
    h, w, _ = out.shape

    # Dark semi-transparent top HUD bar
    banner_h = 58
    out[:banner_h, :] = (out[:banner_h, :].astype(np.float32) * 0.25).astype(np.uint8)

    # Progress bar along the very top (4px height)
    prog_w = int(np.clip(progress_ratio, 0.0, 1.0) * w)
    out[:4, :prog_w] = (40, 220, 120)

    # Draw main title & subtitle
    draw_text(out, stage_title, x=14, y=12, scale=2, color=(255, 230, 80))
    draw_text(out, subtitle, x=14, y=34, scale=2, color=(220, 240, 255))

    # Draw red alert badge when external perturbation force is active
    if is_push_active:
        badge_w = 185
        out[12:44, w - badge_w - 12 : w - 12] = (210, 35, 35)
        draw_text(out, "EXTERNAL PUSH!", x=w - badge_w - 2, y=21, scale=2, color=(255, 255, 255))

    return out


def discover_progression_stages(checkpoint_dir: str):
    """
    Automatically find saved checkpoints in `./checkpoints/` and assemble a clean
    chronological progression from Step 0 (Untrained) to Final (Bait Hunting).
    """
    stages = [
        {
            "label": "STAGE 1: STEP 0 (UNTRAINED RANDOM POLICY)",
            "path": None,
            "use_bait": False,
            "push_force": 30.0,
            "untrained": True,
        }
    ]

    # Find numbered checkpoints: quad_ppo_perturb_<N>_steps.zip and quad_ppo_bait_<N>_steps.zip
    def _extract_steps(filepath):
        m = re.search(r"_(\d+)_steps\.zip$", os.path.basename(filepath))
        return int(m.group(1)) if m else 0

    perturb_ckpts = sorted(
        glob.glob(os.path.join(checkpoint_dir, "quad_ppo_perturb_*_steps.zip")),
        key=_extract_steps,
    )
    bait_ckpts = sorted(
        glob.glob(os.path.join(checkpoint_dir, "quad_ppo_bait_*_steps.zip")),
        key=_extract_steps,
    )

    # Pick up to 3 evenly spaced checkpoints from the main locomotion training
    if perturb_ckpts:
        indices = np.unique(np.linspace(0, len(perturb_ckpts) - 1, min(3, len(perturb_ckpts)), dtype=int))
        for idx in indices:
            ckpt_path = perturb_ckpts[idx]
            steps = _extract_steps(ckpt_path)
            stages.append({
                "label": f"TRAINING: {steps:,} STEPS (GAIT LEARNING)",
                "path": ckpt_path,
                "use_bait": False,
                "push_force": 45.0,
                "untrained": False,
            })

    # Add the 3M final locomotion model if present
    final_loco = os.path.join(checkpoint_dir, "final_model.zip")
    if os.path.exists(final_loco):
        stages.append({
            "label": "3,000,000 STEPS (ROBUST PUSH RECOVERY)",
            "path": final_loco,
            "use_bait": False,
            "push_force": 65.0,
            "untrained": False,
        })

    # Add intermediate bait checkpoint if present
    if bait_ckpts:
        mid_bait = bait_ckpts[len(bait_ckpts) // 2]
        b_steps = _extract_steps(mid_bait)
        stages.append({
            "label": f"BAIT FINE-TUNING: {b_steps:,} STEPS (360 TURN)",
            "path": mid_bait,
            "use_bait": True,
            "push_force": 50.0,
            "untrained": False,
        })

    # Add the final best/bait model with 360-degree random cell bait hunting
    final_bait = os.path.join(checkpoint_dir, "final_bait_model.zip")
    best_model = os.path.join(checkpoint_dir, "best_model", "best_model.zip")
    chosen_final = final_bait if os.path.exists(final_bait) else (best_model if os.path.exists(best_model) else None)

    if chosen_final:
        stages.append({
            "label": "FINAL POLICY: 360 BAIT HUNTING + PUSH RECOVERY",
            "path": chosen_final,
            "use_bait": True,
            "push_force": 60.0,
            "untrained": False,
        })

    return stages


def record_progression_video(
    checkpoint_dir="./checkpoints",
    output_video="training_progression_0_to_final.mp4",
    steps_per_stage=250,
):
    stages = discover_progression_stages(checkpoint_dir)
    num_stages = len(stages)
    print("=" * 74)
    print(f"GENERATING TRAINING PROGRESSION VIDEO ({num_stages} STAGES, {steps_per_stage} STEPS/STAGE)")
    for i, st in enumerate(stages, 1):
        print(f"  [{i}/{num_stages}] {st['label']} -> {st['path'] or 'Random Untrained Weights'}")
    print("=" * 74)

    width, height = 640, 480
    all_frames = []

    for stage_idx, st in enumerate(stages):
        is_Final_Bait = st["use_bait"]
        env = QuadrupedPerturbationEnv(
            perturbation_prob=0.025,
            max_push_force=st["push_force"],
            push_duration_steps=6,
            use_bait=is_Final_Bait,
        )

        renderer = mujoco.Renderer(env.model, height=height, width=width)
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        camera.trackbodyid = env.trunk_body_id
        camera.distance = 3.5 if is_Final_Bait else 2.4
        camera.elevation = -30 if is_Final_Bait else -20
        camera.azimuth = 115 if is_Final_Bait else 90

        model = None
        if st["path"] is not None and os.path.exists(st["path"]):
            model = PPO.load(st["path"])

        obs, _ = env.reset(seed=100 + stage_idx)
        if not is_Final_Bait:
            env.command = np.array([0.85, 0.0, 0.0], dtype=np.float32)
            obs = env._get_obs()

        falls = 0
        stage_steps = steps_per_stage + (150 if stage_idx == num_stages - 1 else 0)

        for step in range(stage_steps):
            if st["untrained"]:
                # Untrained step-0 random exploration policy with exaggerated noise
                action = np.random.uniform(-1.0, 1.0, size=env.num_joints).astype(np.float32)
                env.data.ctrl[:12] = env.default_qpos + action * 0.65
                for _ in range(env.sim_substeps):
                    mujoco.mj_step(env.model, env.data)
                trunk_h = float(env.data.qpos[2])
                terminated = trunk_h < 0.17 or trunk_h > 0.45
                info = {"is_push_active": False}
            else:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)

            renderer.update_scene(env.data, camera=camera)
            raw_frame = renderer.render()

            vx = float(env.data.qvel[0])
            rx, ry = float(env.data.qpos[0]), float(env.data.qpos[1])
            if is_Final_Bait:
                subtitle = f"POS: ({rx:.1f}M, {ry:.1f}M) | BAITS COLLECTED: {env.baits_collected} | FALLS: {falls}"
            else:
                subtitle = f"DIST X: {rx:.1f}M | SPEED: {vx:.2f} M/S | FALLS: {falls}"

            progress = (stage_idx + (step / stage_steps)) / num_stages
            annotated = annotate_frame(
                raw_frame,
                stage_title=st["label"],
                subtitle=subtitle,
                progress_ratio=progress,
                is_push_active=bool(info.get("is_push_active", False)),
            )
            all_frames.append(annotated)

            if terminated:
                falls += 1
                obs, _ = env.reset()
                if not is_Final_Bait:
                    env.command = np.array([0.85, 0.0, 0.0], dtype=np.float32)
                    obs = env._get_obs()

        renderer.close()
        print(f"[DONE] Stage {stage_idx + 1}/{num_stages}: {st['label']} (Falls: {falls})")

    print(f"[INFO] Saving progression video ({len(all_frames)} frames at 50 FPS) to: {output_video}")
    imageio.mimsave(output_video, all_frames, fps=50)
    print(f"[SUCCESS] Full training progression video saved: {output_video}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Record Step 0 to Final Training Progression Video")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints")
    parser.add_argument("--output", type=str, default="training_progression_0_to_final.mp4")
    parser.add_argument("--steps-per-stage", type=int, default=250, help="Frames per checkpoint stage (50 FPS = 5s)")
    args = parser.parse_args()

    record_progression_video(args.checkpoint_dir, args.output, args.steps_per_stage)
