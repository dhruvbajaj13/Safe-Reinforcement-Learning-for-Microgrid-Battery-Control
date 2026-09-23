"""
evaluate.py
-----------
Loads the two trained agents (safe / unsafe) and runs each on a *fresh* test
week of data (different random seed than training, so it's a genuine
out-of-sample test). Produces:
  - results/comparison.png   (SOC trajectory, grid cost, violations)
  - printed summary table in the terminal

Usage:
    python evaluate.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

from microgrid_env import make_env

MODELS_DIR = "models"
RESULTS_DIR = "results"
TEST_SEED = 999   # different from training seed -> unseen data
TEST_DAYS = 7


def run_episode(model_path: str, use_safety_shield: bool):
    env = make_env(num_days=TEST_DAYS, use_safety_shield=use_safety_shield, seed=TEST_SEED)
    model = PPO.load(model_path)

    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    return pd.DataFrame(env.history)


def summarize(df: pd.DataFrame, label: str):
    total_cost = df["cost"].sum()
    num_violations = int(df["violation"].sum())
    num_clipped = int(df["was_clipped"].sum())
    min_soc, max_soc = df["soc"].min(), df["soc"].max()
    print(f"\n--- {label} ---")
    print(f"Total grid cost over {TEST_DAYS} days : {total_cost:.2f}")
    print(f"Safety violations (SOC out of range) : {num_violations}")
    print(f"Steps where shield corrected action   : {num_clipped}")
    print(f"SOC range reached                     : {min_soc:.3f} - {max_soc:.3f}")
    return {
        "label": label,
        "total_cost": total_cost,
        "violations": num_violations,
        "clipped_steps": num_clipped,
        "min_soc": min_soc,
        "max_soc": max_soc,
    }


def plot_comparison(df_safe: pd.DataFrame, df_unsafe: pd.DataFrame):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)

    axes[0].plot(df_safe["step"], df_safe["soc"], label="Safe RL (shield ON)")
    axes[0].plot(df_unsafe["step"], df_unsafe["soc"], label="Plain RL (shield OFF)", alpha=0.8)
    axes[0].axhline(0.10, color="red", linestyle="--", linewidth=1, label="Safety bounds")
    axes[0].axhline(0.95, color="red", linestyle="--", linewidth=1)
    axes[0].set_ylabel("Battery SOC")
    axes[0].legend(loc="upper right")
    axes[0].set_title("Battery state-of-charge over the test week")

    axes[1].plot(df_safe["step"], df_safe["cost"].cumsum(), label="Safe RL (shield ON)")
    axes[1].plot(df_unsafe["step"], df_unsafe["cost"].cumsum(), label="Plain RL (shield OFF)")
    axes[1].set_ylabel("Cumulative grid cost")
    axes[1].legend(loc="upper left")
    axes[1].set_title("Cumulative electricity cost bought from the grid")

    violation_steps_unsafe = df_unsafe[df_unsafe["violation"]]["step"]
    axes[2].scatter(violation_steps_unsafe, np.ones(len(violation_steps_unsafe)),
                     color="red", label="Unsafe agent violation", marker="x")
    axes[2].set_ylim(0.5, 1.5)
    axes[2].set_yticks([])
    axes[2].set_xlabel("Timestep (hour)")
    axes[2].set_title("Safety violations (plain RL only — safe RL should have none)")
    axes[2].legend(loc="upper right")

    fig.tight_layout()
    out_path = os.path.join(RESULTS_DIR, "comparison.png")
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved plot to {out_path}")


def main():
    safe_path = os.path.join(MODELS_DIR, "ppo_safe.zip")
    unsafe_path = os.path.join(MODELS_DIR, "ppo_unsafe.zip")

    if not (os.path.exists(safe_path) and os.path.exists(unsafe_path)):
        raise FileNotFoundError(
            "Trained models not found. Run `python train.py` first "
            "(it saves models/ppo_safe.zip and models/ppo_unsafe.zip)."
        )

    df_safe = run_episode(safe_path, use_safety_shield=True)
    df_unsafe = run_episode(unsafe_path, use_safety_shield=False)

    summary_safe = summarize(df_safe, "SAFE RL (shield ON)")
    summary_unsafe = summarize(df_unsafe, "PLAIN RL (shield OFF)")

    plot_comparison(df_safe, df_unsafe)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    pd.DataFrame([summary_safe, summary_unsafe]).to_csv(
        os.path.join(RESULTS_DIR, "summary.csv"), index=False
    )
    print(f"Saved summary table to {RESULTS_DIR}/summary.csv")


if __name__ == "__main__":
    main()
