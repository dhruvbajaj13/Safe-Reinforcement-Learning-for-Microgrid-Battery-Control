"""
microgrid_env.py
-----------------
A small, fully custom Gymnasium environment simulating a home/building with:
  - solar panels (variable, weather-dependent generation)
  - a battery (the thing the RL agent controls)
  - a household demand profile
  - a grid connection (buys any shortfall, at a price)

This plays the same role that the OpenModelica-FMU simulation + OMG wrapper
play in the original research repo -- except it's plain Python, so you can
read every line and explain it in your viva.

Action space:  Box(-1, 1) -> -1 = discharge battery fully, +1 = charge fully
Observation:   [soc, solar_kw, demand_kw, sin(hour), cos(hour)]
Reward:        -(cost of electricity bought from the grid) - (safety penalty,
                only relevant when the shield is turned OFF)
"""

from dataclasses import dataclass
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from safety_shield import BatteryParams, shield_action
from data_utils import generate_synthetic_profile


@dataclass
class EnvConfig:
    grid_price_per_kwh: float = 8.0     # e.g. Rs/kWh — tune to your currency
    unsafe_penalty: float = 50.0        # big penalty per step spent outside safe SOC (only if shield disabled)
    use_safety_shield: bool = True      # <-- THE key switch for "safe RL" vs "plain RL"
    battery: BatteryParams = None

    def __post_init__(self):
        if self.battery is None:
            self.battery = BatteryParams()


class MicrogridEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, profile_df, config: EnvConfig = None):
        super().__init__()
        self.profile = profile_df.reset_index(drop=True)
        self.config = config or EnvConfig()
        self.max_steps = len(self.profile) - 1

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 10.0, 10.0, 1.0, 1.0], dtype=np.float32),
        )

        self.step_idx = 0
        self.soc = 0.5
        # episode-level bookkeeping used later for plots/reports
        self.history = []

    def _get_obs(self):
        row = self.profile.iloc[self.step_idx]
        hour = row["hour_of_day"]
        return np.array([
            self.soc,
            row["solar_kw"],
            row["demand_kw"],
            np.sin(2 * np.pi * hour / 24),
            np.cos(2 * np.pi * hour / 24),
        ], dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.step_idx = 0
        self.soc = 0.5
        self.history = []
        return self._get_obs(), {}

    def step(self, action):
        action = float(np.clip(action[0], -1.0, 1.0))
        was_clipped = False

        if self.config.use_safety_shield:
            action, was_clipped = shield_action(self.soc, action, self.config.battery)

        # --- battery physics ---
        params = self.config.battery
        power_battery_kw = action * params.max_power_kw
        if power_battery_kw >= 0:
            delta_soc = (power_battery_kw * params.efficiency * params.dt_hours) / params.capacity_kwh
        else:
            delta_soc = (power_battery_kw / params.efficiency * params.dt_hours) / params.capacity_kwh
        next_soc = self.soc + delta_soc

        # a violation = SOC left the safe window (should basically never happen with the shield ON)
        violation = next_soc < params.soc_min or next_soc > params.soc_max
        next_soc = float(np.clip(next_soc, 0.0, 1.0))  # keep physically valid regardless

        # --- power balance: grid supplies whatever solar + battery discharge can't cover ---
        row = self.profile.iloc[self.step_idx]
        net_load_kw = row["demand_kw"] - row["solar_kw"] + power_battery_kw
        grid_power_kw = max(net_load_kw, 0.0)   # excess solar is simply curtailed (kept simple on purpose)

        cost = grid_power_kw * self.config.grid_price_per_kwh
        reward = -cost
        if violation and not self.config.use_safety_shield:
            reward -= self.config.unsafe_penalty

        self.history.append({
            "step": self.step_idx,
            "soc": self.soc,
            "action": action,
            "was_clipped": was_clipped,
            "violation": violation,
            "grid_power_kw": grid_power_kw,
            "cost": cost,
        })

        self.soc = next_soc
        self.step_idx += 1
        terminated = self.step_idx >= self.max_steps
        truncated = False

        return self._get_obs(), reward, terminated, truncated, {"violation": violation, "was_clipped": was_clipped}


def make_env(num_days=30, use_safety_shield=True, seed=42):
    """Convenience factory used by train.py / evaluate.py."""
    df = generate_synthetic_profile(num_days=num_days, seed=seed)
    config = EnvConfig(use_safety_shield=use_safety_shield)
    return MicrogridEnv(df, config)


if __name__ == "__main__":
    # quick smoke test: random actions for a few steps
    env = make_env(num_days=1)
    obs, _ = env.reset()
    print("Initial obs:", obs)
    for _ in range(5):
        a = env.action_space.sample()
        obs, r, term, trunc, info = env.step(a)
        print(f"action={a[0]:+.2f} soc={obs[0]:.3f} reward={r:.2f} info={info}")
