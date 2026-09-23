"""
data_utils.py
-------------
Generates a synthetic solar-generation + household-demand profile so you don't
need to download any dataset to get started. If you later want to use a real
dataset (e.g. a Kaggle household load / solar dataset), just replace
`generate_synthetic_profile()` with a function that returns the same kind of
pandas DataFrame (columns: "solar_kw", "demand_kw") and everything else in the
project keeps working unchanged.
"""

import numpy as np
import pandas as pd


def generate_synthetic_profile(num_days: int = 30, seed: int = 42, steps_per_day: int = 24) -> pd.DataFrame:
    """
    Builds an hourly profile for `num_days` days.

    - Solar: a bell curve centered at midday (zero at night), scaled 0-5 kW,
      with some day-to-day cloud-cover randomness.
    - Demand: two peaks (morning + evening, like a typical household),
      with random noise so it's not perfectly repetitive.

    Returns a DataFrame with columns: hour_of_day, solar_kw, demand_kw
    """
    rng = np.random.default_rng(seed)
    total_steps = num_days * steps_per_day
    hours = np.arange(total_steps) % steps_per_day

    # --- Solar: bell curve peaking at hour 13, zero before 6am / after 19h ---
    solar = 5.0 * np.exp(-((hours - 13) ** 2) / (2 * 3.0 ** 2))
    solar[(hours < 6) | (hours > 19)] = 0.0
    # day-to-day cloud cover: multiply each day by a random factor 0.4-1.0
    cloud_factor = rng.uniform(0.4, 1.0, size=num_days).repeat(steps_per_day)
    solar = solar * cloud_factor
    solar += rng.normal(0, 0.05, size=total_steps)
    solar = np.clip(solar, 0, None)

    # --- Demand: baseline + morning peak (7-9h) + evening peak (18-22h) ---
    demand = 1.0 + np.zeros(total_steps)
    demand += 2.0 * np.exp(-((hours - 8) ** 2) / (2 * 1.5 ** 2))
    demand += 3.0 * np.exp(-((hours - 20) ** 2) / (2 * 2.0 ** 2))
    demand += rng.normal(0, 0.15, size=total_steps)
    demand = np.clip(demand, 0.2, None)

    df = pd.DataFrame({
        "hour_of_day": hours,
        "solar_kw": solar,
        "demand_kw": demand,
    })
    return df


if __name__ == "__main__":
    # quick manual check: print a day's worth of values
    df = generate_synthetic_profile(num_days=1)
    print(df)
