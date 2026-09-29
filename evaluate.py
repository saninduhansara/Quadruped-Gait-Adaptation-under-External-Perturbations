"""
Evaluation and Push-Recovery Benchmarking for Quadruped Gait Adaptation.
Tests the policy under increasing external lateral and longitudinal push forces.
"""
import os
import argparse
import numpy as np
import torch
from stable_baselines3 import PPO

from quadruped_env import QuadrupedPerturbationEnv

def run_push_benchmark(model_path, push_forces=[0, 20, 40, 60, 80, 100, 120], episodes_per_force=5):
    print("=" * 78)
    print(f"EVALUATING MODEL: {model_path}")
    print("=" * 78)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = PPO.load(model_path)

    results = []

    for force in push_forces:
        env = QuadrupedPerturbationEnv(
            perturbation_prob=0.03,
            max_push_force=force,
            push_duration_steps=6
        )

        survived_episodes = 0
        total_rewards = []
        episode_lengths = []
        distances = []

        for ep in range(episodes_per_force):
            obs, _ = env.reset(seed=1000 + ep)
            env.command = np.array([0.85, 0.0, 0.0], dtype=np.float32)
            obs = env._get_obs()
            ep_reward = 0.0
            steps = 0
            done = False

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                ep_reward += reward
                steps += 1
                done = terminated or truncated

            if not terminated:
                survived_episodes += 1

            total_rewards.append(ep_reward)
            episode_lengths.append(steps)
            distances.append(float(env.data.qpos[0]))

        survival_rate = (survived_episodes / episodes_per_force) * 100.0
        avg_reward = np.mean(total_rewards)
        avg_steps = np.mean(episode_lengths)
        avg_dist = np.mean(distances)

        results.append({
            "force_N": force,
            "survival_rate": survival_rate,
            "avg_reward": avg_reward,
            "avg_steps": avg_steps,
            "avg_dist_m": avg_dist
        })

        print(f"Force: {force:>3d} N | Survival: {survival_rate:>5.1f}% | Avg Dist: {avg_dist:>5.2f} m | Avg Steps: {avg_steps:>6.1f} | Return: {avg_reward:>7.1f}")

    print("=" * 78)
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Quadruped Push Recovery")
    parser.add_argument("--model-path", type=str, default="./checkpoints/best_model/best_model.zip", help="Path to trained model .zip")
    args = parser.parse_args()
    run_push_benchmark(args.model_path)

