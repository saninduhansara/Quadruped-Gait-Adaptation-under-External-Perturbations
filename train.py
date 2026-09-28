"""
PPO Training script for Quadruped Gait Adaptation under External Perturbations.
Optimized for multi-core GPU servers (Ada HPC).
"""
import os
import argparse
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

from setup_robot import setup_robot_model
from quadruped_env import QuadrupedPerturbationEnv

def make_env(rank, seed=0, perturbation_prob=0.02, max_push_force=150.0):
    def _init():
        env = QuadrupedPerturbationEnv(
            perturbation_prob=perturbation_prob,
            max_push_force=max_push_force,
        )
        env.reset(seed=seed + rank)
        return env
    return _init

def train():
    parser = argparse.ArgumentParser(description="Train Quadruped RL Policy with Perturbations")
    parser.add_argument("--num-envs", type=int, default=8, help="Number of parallel simulation environments")
    parser.add_argument("--total-timesteps", type=int, default=10_000_000, help="Total training steps")
    parser.add_argument("--push-force", type=float, default=150.0, help="Maximum perturbation force (N)")
    parser.add_argument("--push-prob", type=float, default=0.02, help="Probability of push per control step")
    parser.add_argument("--save-freq", type=int, default=100_000, help="Save checkpoint every N steps")
    parser.add_argument("--log-dir", type=str, default="./logs", help="Directory for logs and tensorboard")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints", help="Directory for model checkpoints")
    parser.add_argument("--device", type=str, default="auto", help="Device: 'cuda', 'cpu', or 'auto'")
    parser.add_argument("--batch-size", type=int, default=256, help="Minibatch size for PPO updates")
    args = parser.parse_args()

    # Ensure robot model is present
    setup_robot_model()

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    print("=" * 60)
    print("QUADRUPED GAIT ADAPTATION TRAINING")
    print(f"Parallel Envs: {args.num_envs}")
    print(f"Total Timesteps: {args.total_timesteps:,}")
    print(f"Max Perturbation Force: {args.push_force} N (prob: {args.push_prob})")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 60)

    # Create vectorized environment
    env_fns = [make_env(i, perturbation_prob=args.push_prob, max_push_force=args.push_force) for i in range(args.num_envs)]
    vec_env = SubprocVecEnv(env_fns)
    vec_env = VecMonitor(vec_env, filename=os.path.join(args.log_dir, "monitor.csv"))

    # Evaluation environment (unvectorized wrapper)
    eval_env = VecMonitor(SubprocVecEnv([make_env(999, perturbation_prob=args.push_prob, max_push_force=args.push_force)]))

    # Neural network policy architecture (2-layer MLP with ELU activations)
    policy_kwargs = dict(
        activation_fn=torch.nn.ELU,
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
    )

    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=max(args.save_freq // args.num_envs, 1000),
        save_path=args.checkpoint_dir,
        name_prefix="quad_ppo_perturb",
        save_replay_buffer=False,
    )

    # Evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(args.checkpoint_dir, "best_model"),
        log_path=args.log_dir,
        eval_freq=max(25_000 // args.num_envs, 1000),
        n_eval_episodes=5,
        deterministic=True,
    )

    # PPO Hyperparameters tuned for quadruped locomotion
    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=3e-4,
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

    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=[checkpoint_callback, eval_callback],
            progress_bar=False,
        )
        final_model_path = os.path.join(args.checkpoint_dir, "final_model.zip")
        model.save(final_model_path)
        print(f"[SUCCESS] Training completed! Model saved to: {final_model_path}")
    finally:
        vec_env.close()
        eval_env.close()

if __name__ == "__main__":
    train()
