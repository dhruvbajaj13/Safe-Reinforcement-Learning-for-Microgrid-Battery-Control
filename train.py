"""
train.py
--------
Trains TWO PPO agents on the same microgrid environment:
  1. "safe"   -> safety shield ON  (agent's actions are always corrected to stay safe)
  2. "unsafe" -> safety shield OFF (plain RL, can violate battery limits, gets penalized)

This safe-vs-unsafe comparison is the main result/demo for your report:
it shows *why* the safety shield matters, not just that it exists.

Usage:
    python train.py --timesteps 100000
"""

import argparse
import os

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from microgrid_env import make_env

MODELS_DIR = "models"


def train_one(use_safety_shield: bool, timesteps: int, num_days: int, seed: int):
    tag = "safe" if use_safety_shield else "unsafe"
    print(f"\n=== Training {tag.upper()} agent for {timesteps} timesteps ===")

    env = make_vec_env(
        lambda: make_env(num_days=num_days, use_safety_shield=use_safety_shield, seed=seed),
        n_envs=1,
    )

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        seed=seed,
    )
    model.learn(total_timesteps=timesteps)

    os.makedirs(MODELS_DIR, exist_ok=True)
    save_path = os.path.join(MODELS_DIR, f"ppo_{tag}.zip")
    model.save(save_path)
    print(f"Saved {tag} model to {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=100_000,
                         help="Training timesteps per agent (start smaller, e.g. 20000, to iterate fast)")
    parser.add_argument("--num-days", type=int, default=30,
                         help="Days of synthetic data to train on (loops if agent trains longer than 1 episode)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_one(use_safety_shield=True, timesteps=args.timesteps, num_days=args.num_days, seed=args.seed)
    train_one(use_safety_shield=False, timesteps=args.timesteps, num_days=args.num_days, seed=args.seed)

    print("\nDone. Now run: python evaluate.py")


if __name__ == "__main__":
    main()
