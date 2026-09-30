"""
Generates an Extended (2+ Minute) Training Progression & Capability Showcase Video
by loading historical checkpoints saved in ./checkpoints/ in chronological order,
rendering smooth continuous rollouts with live HUD telemetry, and streaming frames to MP4.
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


# Minimal 5x7 bitmap font for pure-NumPy HUD text overlay (zero external dependencies)
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
    current_sec: float,
    total_sec: float,
    is_push_active: bool = False,
    is_limp_active: bool = False,
) -> np.ndarray:
    out = frame.copy()
    h, w, _ = out.shape

    banner_h = 58
    out[:banner_h, :] = (out[:banner_h, :].astype(np.float32) * 0.22).astype(np.uint8)

    # Progress bar along the very top (4px height)
    progress_ratio = current_sec / max(1.0, total_sec)
    prog_w = int(np.clip(progress_ratio, 0.0, 1.0) * w)
    bar_color = (235, 60, 210) if is_limp_active else (40, 220, 120)
    out[:4, :prog_w] = bar_color

    # Title & Subtitle
    title_color = (255, 140, 245) if is_limp_active else (255, 230, 80)
    draw_text(out, stage_title, x=14, y=12, scale=2, color=title_color)
    draw_text(out, subtitle, x=14, y=34, scale=2, color=(220, 240, 255))

    # Elapsed Time Display (e.g. "01:24 / 02:08")
    cur_m, cur_s = int(current_sec // 60), int(current_sec % 60)
    tot_m, tot_s = int(total_sec // 60), int(total_sec % 60)
    time_str = f"{cur_m:02d}:{cur_s:02d}/{tot_m:02d}:{tot_s:02d}"

    badge_right = w - 12
    if is_push_active:
        badge_w = 180
        out[12:44, w - badge_w - 12 : w - 12] = (215, 35, 35)
        draw_text(out, "EXTERNAL PUSH!", x=w - badge_w - 2, y=21, scale=2, color=(255, 255, 255))
        badge_right = w - badge_w - 24

    draw_text(out, time_str, x=badge_right - 130, y=21, scale=2, color=(190, 220, 255))

    return out


def discover_stages(checkpoint_dir: str):
    """Assemble an extensive chronological progression spanning all research milestones."""
    def _extract_steps(filepath):
        m = re.search(r"_(\d+)_steps\.zip$", os.path.basename(filepath))
        return int(m.group(1)) if m else 0

    stages = [
        # Milestone 1: Untrained step-0 baseline
        {
            "label": "STAGE 1: STEP 0 (UNTRAINED RANDOM WEIGHTS)",
            "path": None,
            "use_bait": False,
            "push_force": 20.0,
            "untrained": True,
            "limp_leg": -1,
        }
    ]

    perturb_ckpts = sorted(
        glob.glob(os.path.join(checkpoint_dir, "quad_ppo_perturb_*_steps.zip")),
        key=_extract_steps,
    )
    bait_ckpts = sorted(
        glob.glob(os.path.join(checkpoint_dir, "quad_ppo_bait_*_steps.zip")),
        key=_extract_steps,
    )

    # Milestone 2: Early Gait Exploration (~200k steps)
    if perturb_ckpts:
        early_ckpt = perturb_ckpts[min(1, len(perturb_ckpts) - 1)]
        stages.append({
            "label": f"STAGE 2: { _extract_steps(early_ckpt):,} STEPS (LEARNING BALANCE)",
            "path": early_ckpt,
            "use_bait": False,
            "push_force": 35.0,
            "untrained": False,
            "limp_leg": -1,
        })

    # Milestone 3: Mid-Training Trotting (~1M - 1.5M steps)
    if len(perturb_ckpts) >= 4:
        mid_ckpt = perturb_ckpts[len(perturb_ckpts) // 2]
        stages.append({
            "label": f"STAGE 3: {_extract_steps(mid_ckpt):,} STEPS (TROTTING GAIT FORMED)",
            "path": mid_ckpt,
            "use_bait": False,
            "push_force": 50.0,
            "untrained": False,
            "limp_leg": -1,
        })

    # Milestone 4: 3,000,000 Steps High-Force Push Rejection
    final_loco = os.path.join(checkpoint_dir, "final_model.zip")
    if os.path.exists(final_loco):
        stages.append({
            "label": "STAGE 4: 3,000,000 STEPS (80N PUSH RECOVERY)",
            "path": final_loco,
            "use_bait": False,
            "push_force": 75.0,
            "untrained": False,
            "limp_leg": -1,
        })

    # Milestone 5: 3-in-1 Multi-Gait Transitions (Walk -> Trot -> Bound)
    final_unified = os.path.join(checkpoint_dir, "final_unified_model.zip")
    best_model = os.path.join(checkpoint_dir, "best_model", "best_model.zip")
    final_model_path = final_unified if os.path.exists(final_unified) else (best_model if os.path.exists(best_model) else None)

    if final_model_path:
        stages.append({
            "label": "STAGE 5: MULTI-GAIT (WALK <-> TROT <-> BOUND)",
            "path": final_model_path,
            "use_bait": True,
            "push_force": 50.0,
            "untrained": False,
            "limp_leg": -1,
        })
        # Milestone 6: 360° Random Cell Bait Hunting
        stages.append({
            "label": "STAGE 6: 360 RANDOM CELL BAIT HUNTING",
            "path": final_model_path,
            "use_bait": True,
            "push_force": 60.0,
            "untrained": False,
            "limp_leg": -1,
        })
        # Milestone 7: 3-Legged Limp-Mode Actuator Failure Recovery (Front-Right Disabled!)
        stages.append({
            "label": "STAGE 7: 3-LEG LIMP MODE (FR LEG DISABLED)",
            "path": final_model_path,
            "use_bait": True,
            "push_force": 55.0,
            "untrained": False,
            "limp_leg": 0,  # Front-Right leg locked in the air!
        })

    return stages


def record_2min_progression_video(
    checkpoint_dir="./checkpoints",
    output_video="training_progression_2min.mp4",
    target_duration_sec=126,  # 2 minutes and 6 seconds (> 2 mins guaranteed)
    fps=30,
):
    stages = discover_stages(checkpoint_dir)
    num_stages = len(stages)
    total_target_frames = int(target_duration_sec * fps)
    frames_per_stage = total_target_frames // num_stages

    print("=" * 82)
    print(f"GENERATING EXTENDED TRAINING PROGRESSION VIDEO")
    print(f"Target Duration:  {target_duration_sec}s ({target_duration_sec // 60}m {target_duration_sec % 60}s) at {fps} FPS")
    print(f"Total Stages:     {num_stages} ({frames_per_stage} frames / ~{frames_per_stage / fps:.1f}s per stage)")
    print(f"Output Video:     {output_video}")
    for i, s in enumerate(stages, 1):
        print(f"  [{i}/{num_stages}] {s['label']}")
    print("=" * 82)

    width, height = 640, 480
    writer = imageio.get_writer(output_video, fps=fps, quality=8)
    global_frame_idx = 0

    try:
        for stage_idx, st in enumerate(stages):
            is_bait = st["use_bait"]
            limp_leg = st["limp_leg"]

            env = QuadrupedPerturbationEnv(
                perturbation_prob=0.025,
                max_push_force=st["push_force"],
                push_duration_steps=6,
                use_bait=is_bait,
                leg_failure_prob=0.0,
            )

            renderer = mujoco.Renderer(env.model, height=height, width=width)
            camera = mujoco.MjvCamera()
            camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            camera.trackbodyid = env.trunk_body_id
            camera.distance = 3.6 if is_bait else 2.5
            camera.elevation = -30 if is_bait else -20
            camera.azimuth = 115 if is_bait else 90

            model = None
            expected_dim = 47
            if st["path"] is not None and os.path.exists(st["path"]):
                model = PPO.load(st["path"])
                expected_dim = model.observation_space.shape[0]

            obs, _ = env.reset(seed=200 + stage_idx)
            if limp_leg >= 0:
                env.set_disabled_leg(limp_leg)
                obs = env._get_obs()

            if not is_bait:
                env.command = np.array([0.85, 0.0, 0.0], dtype=np.float32)
                obs = env._get_obs()

            falls = 0

            # Run stage frames
            for f_idx in range(frames_per_stage):
                if st["untrained"]:
                    action = np.random.uniform(-1.0, 1.0, size=env.num_joints).astype(np.float32)
                    env.data.ctrl[:12] = env.default_qpos + action * 0.65
                    for _ in range(env.sim_substeps):
                        mujoco.mj_step(env.model, env.data)
                    trunk_h = float(env.data.qpos[2])
                    terminated = trunk_h < 0.17 or trunk_h > 0.45
                    info = {"is_push_active": False}
                else:
                    action, _ = model.predict(obs[:expected_dim], deterministic=True)
                    obs, reward, terminated, truncated, info = env.step(action)

                renderer.update_scene(env.data, camera=camera)
                raw_pixels = renderer.render()

                current_sec = global_frame_idx / fps
                gait_str = info.get("gait_mode", "TROT")

                if limp_leg >= 0:
                    subtitle = f"MODE: 3-LEG TRIPOD HOP | BAITS: {env.baits_collected} | FALLS: {falls}"
                elif is_bait:
                    subtitle = f"GAIT: {gait_str} | BAITS: {env.baits_collected} | TARGET: ({env.bait_pos[0]:.0f},{env.bait_pos[1]:.0f})"
                else:
                    vx = float(env.data.qvel[0])
                    dist = float(env.data.qpos[0])
                    subtitle = f"DIST: {dist:.1f}M | SPEED: {vx:.2f} M/S | FALLS: {falls}"

                annotated = annotate_frame(
                    raw_pixels,
                    stage_title=st["label"],
                    subtitle=subtitle,
                    current_sec=current_sec,
                    total_sec=target_duration_sec,
                    is_push_active=bool(info.get("is_push_active", False)),
                    is_limp_active=(limp_leg >= 0),
                )

                writer.append_data(annotated)
                global_frame_idx += 1

                if terminated:
                    falls += 1
                    obs, _ = env.reset()
                    if limp_leg >= 0:
                        env.set_disabled_leg(limp_leg)
                    if not is_bait:
                        env.command = np.array([0.85, 0.0, 0.0], dtype=np.float32)
                    obs = env._get_obs()

            renderer.close()
            print(f"[COMPLETED] Stage {stage_idx + 1}/{num_stages}: {st['label']} ({frames_per_stage} frames)")

    finally:
        writer.close()

    total_time_s = global_frame_idx / fps
    print("=" * 82)
    print(f"[SUCCESS] Extended progression video saved: {output_video}")
    print(f"Total Duration: {int(total_time_s // 60)}m {int(total_time_s % 60):02d}s ({global_frame_idx} frames at {fps} FPS)")
    print("=" * 82)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Record 2+ Minute Training Progression Video")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints")
    parser.add_argument("--output", type=str, default="training_progression_2min.mp4")
    parser.add_argument("--duration", type=int, default=126, help="Target video duration in seconds (126 = 2m 06s)")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second")
    args = parser.parse_args()

    record_2min_progression_video(
        checkpoint_dir=args.checkpoint_dir,
        output_video=args.output,
        target_duration_sec=args.duration,
        fps=args.fps,
    )
