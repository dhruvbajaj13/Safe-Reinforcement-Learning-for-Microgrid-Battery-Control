"""
frontend/build_dashboard.py
----------------------------
Regenerates the data shown in dashboard.html:
  1. Runs the trained "safe" PPO agent (shield ON) on the test week.
  2. Runs a random/naive policy with the shield OFF, as the honest
     "what happens without protection" baseline.
  3. Injects both trajectories into dashboard_template.html and writes
     the final dashboard.html (fully self-contained, no build step needed
     to view it -- just open it in a browser).

Run this again any time you retrain the safe model, so the dashboard
reflects your latest results:

    cd safe_rl_microgrid
    python train.py --timesteps 100000
    python frontend/build_dashboard.py
"""

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluate import run_episode, MODELS_DIR, TEST_DAYS, TEST_SEED  # noqa: E402
from microgrid_env import make_env  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def compact(df: pd.DataFrame) -> dict:
    return {
        "step": df["step"].tolist(),
        "soc": [round(x, 4) for x in df["soc"].tolist()],
        "cum_cost": [round(x, 2) for x in df["cost"].cumsum().tolist()],
        "violation": df["violation"].astype(int).tolist(),
        "clipped": df["was_clipped"].astype(int).tolist(),
    }


def main():
    safe_path = os.path.join(os.path.dirname(HERE), MODELS_DIR, "ppo_safe.zip")
    if not os.path.exists(safe_path):
        raise FileNotFoundError("Run `python train.py` first so models/ppo_safe.zip exists.")

    df_safe = run_episode(safe_path, use_safety_shield=True)

    rng = np.random.default_rng(7)
    env = make_env(num_days=TEST_DAYS, use_safety_shield=False, seed=TEST_SEED)
    obs, _ = env.reset()
    done = False
    while not done:
        action = rng.uniform(-1, 1, size=(1,)).astype("float32")
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    df_unsafe = pd.DataFrame(env.history)

    data = {"safe": compact(df_safe), "unsafe": compact(df_unsafe)}
    data_str = json.dumps(data)

    template_path = os.path.join(HERE, "dashboard_template.html")
    with open(template_path, "r", encoding="utf-8") as f:
        html = f.read()
    html = html.replace("__DEMO_DATA__", data_str)

    out_path = os.path.join(HERE, "dashboard.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Safe agent   -> cost {df_safe['cost'].sum():.1f}, violations {int(df_safe['violation'].sum())}")
    print(f"Random policy -> cost {df_unsafe['cost'].sum():.1f}, violations {int(df_unsafe['violation'].sum())}")
    print(f"Wrote {out_path} -- open it directly in any browser, no server needed.")


if __name__ == "__main__":
    main()
