"""
PPO Training & Warm-Start Retraining Pipeline for the Unified Research-Grade Quadruped:
  1. 360° Random Cell Bait Navigation
  2. 3-in-1 Multi-Gait Controller (Walk -> Trot -> Bound/Gallop)
  3. 3-Legged Limp-Mode Adaptation (Actuator Failure Recovery)
  4. Proprioceptive History / RMA (Rapid Motor Adaptation) 10-Step Temporal Encoding
Optimized for multi-core GPU servers (Ada HPC).
"""
import os
import argparse
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback

from setup_robot import setup_robot_model
from quadruped_env import QuadrupedPerturbationEnv


class AdaptiveCurriculumCallback(BaseCallback):
    """
    Gradually ramps up both:
      1. External perturbation push force (0 N -> max_force)
      2. 3-Legged actuator failure probability (0.0 -> max_fault_prob)
    so the policy masters multi-gait bait hunting and smoothly learns 3-legged limp recovery.
    """
    def __init__(
        self,
        vec_env,
        eval_env=None,
        max_force=60.0,
        max_fault_prob=0.25,
        warmup_steps=100_000,
        ramp_steps=600_000,
        verbose=1,
    ):
        super().__init__(verbose)
        self.vec_env = vec_env
        self.eval_env = eval_env
        self.max_force = max_force
        self.max_fault_prob = max_fault_prob
        self.warmup_steps = warmup_steps
        self.ramp_steps = ramp_steps
        self.last_force = -1.0

    def _on_step(self) -> bool:
        if self.n_calls % 200 == 0:
            if self.num_timesteps < self.warmup_steps:
                progress = 0.0
            else:
                progress = min(1.0, (self.num_timesteps - self.warmup_steps) / max(1, self.ramp_steps))

            current_force = progress * self.max_force
            current_fault_prob = progress * self.max_fault_prob

            if abs(current_force - self.last_force) >= 2.0 or current_force == self.max_force:
                self.vec_env.env_method("set_max_push_force", current_force)
                self.vec_env.env_method("set_leg_failure_prob", current_fault_prob)
                if self.eval_env is not None:
                    self.eval_env.env_method("set_max_push_force", current_force)
                    self.eval_env.env_method("set_leg_failure_prob", current_fault_prob * 0.5)
                if abs(current_force - self.last_force) >= 10.0 or self.last_force < 0:
                    print(
                        f"[CURRICULUM] Step {self.num_timesteps:,}: "
                        f"Push Force -> {current_force:.1f} N | 3-Leg Fault Prob -> {current_fault_prob * 100:.1f}%"
                    )
                self.last_force = current_force
        return True


def transfer_weights_47_to_78(new_model: PPO, old_ckpt_path: str, device: str = "auto") -> bool:
    """
    Warm-start surgery: Transfers trained weights from a 47-dim checkpoint into the
    expanded 78-dim (Multi-Gait + 3-Leg Health + RMA History) policy network.
    If the checkpoint is already 78-dim, loads it directly.
    """
    old_model = PPO.load(old_ckpt_path, device=device)
    old_obs_dim = old_model.observation_space.shape[0]
    new_obs_dim = new_model.observation_space.shape[0]

    if old_obs_dim == new_obs_dim:
        new_model.policy.load_state_dict(old_model.policy.state_dict())
        print(f"[WARM-START] Loaded full {new_obs_dim}-dim policy weights from: {old_ckpt_path}")
        return True

    if old_obs_dim == 47 and new_obs_dim > 47:
        with torch.no_grad():
            old_sd = old_model.policy.state_dict()
            new_sd = new_model.policy.state_dict()

            for key in new_sd.keys():
                if key not in old_sd:
                    continue
                if new_sd[key].shape == old_sd[key].shape:
                    new_sd[key].copy_(old_sd[key])
                elif new_sd[key].ndim == 2 and new_sd[key].shape[0] == old_sd[key].shape[0] and old_sd[key].shape[1] == 47:
                    # First MLP layer: copy the 47 core proprioception columns and initialize the new 31 columns near zero
                    new_sd[key][:, :47].copy_(old_sd[key])
                    new_sd[key][:, 47:].mul_(0.05)

            new_model.policy.load_state_dict(new_sd)
        print(
            f"[WARM-START] Transferred 47-dim locomotion weights from '{old_ckpt_path}' "
            f"into expanded {new_obs_dim}-dim (Multi-Gait + 3-Leg Limp + RMA History) policy!"
        )
        return True

    print(f"[WARN] Could not transfer weights from {old_obs_dim}-dim checkpoint to {new_obs_dim}-dim policy.")
    return False


def make_env(rank, seed=0, perturbation_prob=0.02, max_push_force=0.0, leg_failure_prob=0.0, use_bait=True):
    def _init():
        env = QuadrupedPerturbationEnv(
            perturbation_prob=perturbation_prob,
            max_push_force=max_push_force,
            leg_failure_prob=leg_failure_prob,
            use_bait=use_bait,
        )
        env.reset(seed=seed + rank)
        return env
    return _init


def train():
    parser = argparse.ArgumentParser(
        description="Train Quadruped RL Policy with 3-in-1 Multi-Gait, 3-Legged Fault Recovery, RMA History & Bait Navigation"
    )
    parser.add_argument("--num-envs", type=int, default=8, help="Number of parallel simulation environments")
    parser.add_argument("--total-timesteps", type=int, default=1_500_000, help="Total training steps")
    parser.add_argument("--push-force", type=float, default=60.0, help="Maximum perturbation force (N)")
    parser.add_argument("--push-prob", type=float, default=0.02, help="Probability of push per control step")
    parser.add_argument("--fault-prob", type=float, default=0.25, help="Max episode probability of 3-legged motor failure")
    parser.add_argument("--resume-from", type=str, default="./checkpoints/best_model/best_model.zip",
                        help="Path to existing 47-dim or 78-dim checkpoint for warm-start retraining")
    parser.add_argument("--no-bait", action="store_true", help="Disable random cell bait mode")
    parser.add_argument("--save-freq", type=int, default=100_000, help="Save checkpoint every N steps")
    parser.add_argument("--log-dir", type=str, default="./logs", help="Directory for logs and tensorboard")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints", help="Directory for model checkpoints")
    parser.add_argument("--device", type=str, default="auto", help="Device: 'cuda', 'cpu', or 'auto'")
    parser.add_argument("--batch-size", type=int, default=256, help="Minibatch size for PPO updates")
    args = parser.parse_args()

    use_bait = not args.no_bait
    scene_xml = setup_robot_model()
    from quadruped_env import _ensure_bait_scene_xml
    _ensure_bait_scene_xml(scene_xml)

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    print("=" * 74)
    print("QUADRUPED UNIFIED RL TRAINING: MULTI-GAIT + 3-LEG LIMP + RMA + BAIT")
    print(f"Parallel Envs:         {args.num_envs}")
    print(f"Total Timesteps:       {args.total_timesteps:,}")
    print(f"360° Bait Navigation:  {'ENABLED' if use_bait else 'DISABLED'}")
    print(f"3-in-1 Multi-Gait:     WALK (0) -> TROT (1) -> BOUND (2)")
    print(f"Max Push Perturbation: {args.push_force} N (prob: {args.push_prob})")
    print(f"Max 3-Leg Fault Prob:  {args.fault_prob * 100:.0f}% of episodes")
    print(f"RMA History Window:    10 steps (0.20s temporal encoding, 78-dim obs)")
    print(f"CUDA Available:        {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Using GPU:             {torch.cuda.get_device_name(0)}")
    print("=" * 74)

    env_fns = [
        make_env(i, perturbation_prob=args.push_prob, max_push_force=0.0, leg_failure_prob=0.0, use_bait=use_bait)
        for i in range(args.num_envs)
    ]
    vec_env = SubprocVecEnv(env_fns)
    vec_env = VecMonitor(vec_env, filename=os.path.join(args.log_dir, "monitor.csv"))

    eval_env = VecMonitor(
        SubprocVecEnv([make_env(999, perturbation_prob=args.push_prob, max_push_force=0.0, leg_failure_prob=0.0, use_bait=use_bait)])
    )

    policy_kwargs = dict(
        activation_fn=torch.nn.ELU,
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(args.save_freq // args.num_envs, 1000),
        save_path=args.checkpoint_dir,
        name_prefix="quad_ppo_unified",
        save_replay_buffer=False,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(args.checkpoint_dir, "best_model"),
        log_path=args.log_dir,
        eval_freq=max(25_000 // args.num_envs, 1000),
        n_eval_episodes=5,
        deterministic=True,
    )

    has_resume = bool(args.resume_from and os.path.exists(args.resume_from))
    lr = 2e-4 if has_resume else 3e-4

    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=lr,
        n_steps=2048,
        batch_size=args.batch_size,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        tensorboard_log=os.path.join(args.log_dir, "tb"),
        verbose=1,
        device=args.device,
    )

    if has_resume:
        transfer_weights_47_to_78(model, args.resume_from, device=args.device)
        warmup = min(75_000, args.total_timesteps // 10)
        ramp = min(500_000, args.total_timesteps // 2)
    else:
        print("[INFO] Starting fresh 78-dim PPO training run...")
        warmup = min(200_000, args.total_timesteps // 6)
        ramp = min(800_000, args.total_timesteps // 2)

    curriculum_callback = AdaptiveCurriculumCallback(
        vec_env,
        eval_env=eval_env,
        max_force=args.push_force,
        max_fault_prob=args.fault_prob,
        warmup_steps=warmup,
        ramp_steps=ramp,
        verbose=1,
    )

    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=[checkpoint_callback, eval_callback, curriculum_callback],
            progress_bar=False,
        )
        final_model_path = os.path.join(args.checkpoint_dir, "final_unified_model.zip")
        model.save(final_model_path)
        print(f"[SUCCESS] Unified Multi-Gait + 3-Leg + RMA Training completed! Saved to: {final_model_path}")
    finally:
        vec_env.close()
        eval_env.close()


if __name__ == "__main__":
    train()
