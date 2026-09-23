"""
dashboard.py
------------
Optional Streamlit demo you can show in your viva. Lets you step through the
test week interactively and see the safe agent's decisions + battery SOC live.

Run with:
    streamlit run dashboard.py

Requires the safe model to already exist (run train.py first).
"""

import streamlit as st
import pandas as pd
from stable_baselines3 import PPO

from microgrid_env import make_env

st.set_page_config(page_title="Safe RL Microgrid Demo", layout="wide")
st.title("Safe Reinforcement Learning — Microgrid Battery Manager")

MODEL_PATH = "models/ppo_safe.zip"


@st.cache_resource
def load_model():
    return PPO.load(MODEL_PATH)


@st.cache_data
def run_full_episode():
    model = load_model()
    env = make_env(num_days=7, use_safety_shield=True, seed=999)
    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    return pd.DataFrame(env.history)


try:
    df = run_full_episode()
except FileNotFoundError:
    st.error("Model not found. Run `python train.py` first to create models/ppo_safe.zip")
    st.stop()

step = st.slider("Timestep (hour)", 0, len(df) - 1, 0)
row = df.iloc[step]

col1, col2, col3, col4 = st.columns(4)
col1.metric("Battery SOC", f"{row['soc']*100:.1f}%")
col2.metric("Action", f"{row['action']:+.2f}")
col3.metric("Grid power drawn (kW)", f"{row['grid_power_kw']:.2f}")
col4.metric("Shield intervened?", "Yes" if row["was_clipped"] else "No")

st.subheader("Battery SOC over the whole test week")
st.line_chart(df.set_index("step")["soc"])

st.subheader("Cumulative grid electricity cost")
st.line_chart(df.set_index("step")["cost"].cumsum())

st.caption(
    "The red dashed safety bounds (10%–95% SOC) are enforced by the rule-based "
    "safety shield before every action reaches the simulated battery."
)
