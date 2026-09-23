# Safe Reinforcement Learning for Microgrid Battery Control (Easy Version)

A beginner-friendly B.Tech project inspired by
[Webbah/safe-reinforcement-learing-for-microgrid-control](https://github.com/Webbah/safe-reinforcement-learing-for-microgrid-control),
rebuilt with a simple pure-Python simulation instead of OpenModelica/FMU, and
a rule-based safety shield instead of polytope/feasible-set math.

## Idea

An RL agent decides how much to charge/discharge a home battery every hour,
to minimize electricity bought from the grid, given solar generation and
household demand. A **safety shield** makes sure the agent's action can never
over-charge or over-discharge the battery, no matter what the RL agent
outputs.

Two agents are trained and compared:
- **Safe RL** — safety shield ON (actions are corrected before they execute)
- **Plain RL** — safety shield OFF (can violate battery limits, gets penalized)

This safe-vs-plain comparison is the core result of the project.

## Files

| File | What it does |
|---|---|
| `data_utils.py` | Generates a synthetic solar + household demand profile (no dataset download needed) |
| `safety_shield.py` | The rule-based safety layer — clips unsafe actions before they reach the battery |
| `microgrid_env.py` | The custom Gymnasium environment (battery + solar + demand + grid) |
| `train.py` | Trains the safe and unsafe PPO agents |
| `evaluate.py` | Runs both trained agents on unseen test data, plots comparison, saves a summary table |
| `dashboard.py` | Optional Streamlit demo for your viva |
<<<<<<< HEAD
=======
| `frontend/dashboard.html` | Standalone HTML+JS dashboard (no server, no CDN dependency) — double-click to open in any browser. Shows the safe-vs-unsafe comparison and a live in-browser simulator you can control. |
| `frontend/build_dashboard.py` | Regenerates `frontend/dashboard.html` with fresh results after you retrain |
>>>>>>> bcf19ab7 (Commited)

## Setup

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
# 1. Train both agents (start small to iterate fast, then increase for final results)
python train.py --timesteps 20000

# 2. Evaluate + generate comparison plots
python evaluate.py
# -> results/comparison.png
# -> results/summary.csv

<<<<<<< HEAD
# 3. (Optional) interactive demo for your viva
streamlit run dashboard.py
=======
# 3. (Optional) interactive Streamlit demo for your viva
streamlit run dashboard.py

# 4. (Optional) standalone HTML dashboard -- no server needed, works offline
python frontend/build_dashboard.py
# then just double-click frontend/dashboard.html, or open it in any browser
>>>>>>> bcf19ab7 (Commited)
```

## What to expect / tune

- With very few timesteps (e.g. 2,000, used only to smoke-test the pipeline)
  the agents won't have learned much yet — the *safety shield still works
  correctly* even from a random/untrained policy, since it's rule-based and
  independent of learning. This is a good thing to point out in your report:
  safety doesn't depend on how well the agent has learned.
- For real results, train with `--timesteps 100000` or more (increase
  `--num-days` too, so the agent sees more varied days). Training time
  depends on your CPU; a few minutes to ~30 minutes is typical for this
  problem size.
- Try changing `grid_price_per_kwh`, `unsafe_penalty`, or the battery's
  `capacity_kwh` / `max_power_kw` in `microgrid_env.py` (`EnvConfig`,
  `BatteryParams`) to see how the agent's behavior changes — good material
  for a "sensitivity analysis" section in your report.

## Extending it (stretch goals)

- Swap `generate_synthetic_profile()` in `data_utils.py` for a real household
  load + solar dataset (same DataFrame columns: `solar_kw`, `demand_kw`).
- Add a second battery, or an EV charging load, to the environment.
- Add a simple day-ahead solar/demand forecast as extra observation features.
- Try DDPG or SAC from Stable-Baselines3 instead of PPO and compare.

## Mapping back to the original research repo

| Original repo (hard) | This project (easy) |
|---|---|
| OpenModelica + FMU (pyfmi) circuit simulation | `microgrid_env.py` — plain Python battery/solar/demand model |
| `openmodelica_microgrid_gym` (OMG) Gym wrapper | `MicrogridEnv(gymnasium.Env)` |
| Polytope / feasible-set safeguard (`safeguard_validation.py`) | `safety_shield.py` — SOC bounds-checking |
| Stable-Baselines3 agent | Same library — PPO on the simpler environment |
| Optuna + MongoDB experiment tracking | Skipped — CSV summary is enough for a course project |
