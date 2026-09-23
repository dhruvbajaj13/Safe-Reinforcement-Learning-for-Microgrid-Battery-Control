# 🔋 Safe Reinforcement Learning for Microgrid Battery Control

A beginner-friendly project that applies **Reinforcement Learning (RL)** to battery energy management in a microgrid.

The project compares a **Safe RL agent** with a **Plain RL agent** and demonstrates how a simple rule-based **Safety Shield** can prevent unsafe battery actions while the RL agent learns to reduce electricity usage from the grid.

Inspired by the research project [Webbah/safe-reinforcement-learing-for-microgrid-control](https://github.com/Webbah/safe-reinforcement-learing-for-microgrid-control), this implementation replaces the complex OpenModelica/FMUs and mathematical feasible-set calculations with a lightweight **pure-Python simulation**.

---

## 🚀 Project Overview

In a home microgrid, electricity can come from:

* ☀️ Solar generation
* 🔌 The main electricity grid
* 🔋 A battery storage system

The RL agent decides whether the battery should:

* Charge
* Discharge
* Stay idle

The objective is to reduce the amount of electricity purchased from the grid while maintaining safe battery operation.

### The key idea

The project trains and compares two agents:

| Agent        | Safety Shield | Unsafe Actions                     |
| ------------ | ------------- | ---------------------------------- |
| **Safe RL**  | ✅ Enabled     | Corrected before execution         |
| **Plain RL** | ❌ Disabled    | Penalized when limits are violated |

The **Safety Shield** checks the proposed RL action before it reaches the battery and clips actions that would cause the battery's State of Charge (SOC) to exceed its safe limits.

---

## 🧠 How It Works

```text
                 ☀️ Solar
                    │
                    ▼
              ┌─────────────┐
              │ Microgrid   │
              │ Environment │
              └──────┬──────┘
                     │
                     ▼
               ┌───────────┐
               │ RL Agent  │
               │   PPO     │
               └─────┬─────┘
                     │
              Proposed Action
                     │
                     ▼
             ┌──────────────┐
             │   Safety     │
             │    Shield    │
             └──────┬───────┘
                    │
          Safe / Corrected Action
                    │
                    ▼
             ┌─────────────┐
             │   Battery   │
             │    SOC      │
             └──────┬──────┘
                    │
                    ▼
                 🔌 Grid
```

At every time step:

1. Solar generation and household demand are observed.
2. The PPO agent proposes a battery action.
3. The Safety Shield checks whether the action is safe.
4. Unsafe actions are clipped to the allowable range.
5. The battery state is updated.
6. A reward is calculated based on grid usage and safety violations.

---

## ✨ Key Features

* 🤖 **PPO-based Reinforcement Learning**
* 🔋 Battery charge/discharge control
* ☀️ Synthetic solar generation
* 🏠 Synthetic household electricity demand
* 🛡️ Rule-based Safety Shield
* 📊 Safe RL vs Plain RL comparison
* 📈 Evaluation plots and summary CSV
* 🌐 Optional Streamlit dashboard
* 💻 Standalone offline HTML dashboard
* 🧪 No external dataset required

---

## 📁 Project Structure

```text
Safe-Reinforcement-Learning-for-Microgrid-Battery-Control/
│
├── data_utils.py
├── safety_shield.py
├── microgrid_env.py
├── train.py
├── evaluate.py
├── dashboard.py
├── requirements.txt
├── README.md
│
├── frontend/
│   ├── dashboard.html
│   └── build_dashboard.py
│
└── results/
    ├── comparison.png
    └── summary.csv
```

### File Description

| File                          | Description                                                        |
| ----------------------------- | ------------------------------------------------------------------ |
| `data_utils.py`               | Generates synthetic solar generation and household demand profiles |
| `safety_shield.py`            | Implements the rule-based battery safety layer                     |
| `microgrid_env.py`            | Custom Gymnasium environment for the microgrid                     |
| `train.py`                    | Trains Safe RL and Plain RL PPO agents                             |
| `evaluate.py`                 | Evaluates trained agents and generates comparison results          |
| `dashboard.py`                | Optional Streamlit dashboard for demonstration                     |
| `frontend/dashboard.html`     | Standalone browser-based dashboard                                 |
| `frontend/build_dashboard.py` | Regenerates the HTML dashboard using latest results                |
| `requirements.txt`            | Python dependencies required to run the project                    |

---

## 🛠️ Tech Stack

**Programming Language**

* Python

**Machine Learning / RL**

* Stable-Baselines3
* PPO (Proximal Policy Optimization)

**Environment**

* Gymnasium
* Custom Python-based microgrid simulator

**Data & Visualization**

* NumPy
* Pandas
* Matplotlib

**Dashboard**

* Streamlit
* HTML
* JavaScript

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/dhruvbajaj13/Safe-Reinforcement-Learning-for-Microgrid-Battery-Control.git

cd Safe-Reinforcement-Learning-for-Microgrid-Battery-Control
```

### 2. Create a virtual environment

#### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

#### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Project

### Step 1 — Train the Agents

For a quick test:

```bash
python train.py --timesteps 20000
```

For more meaningful results:

```bash
python train.py --timesteps 100000
```

Training produces the trained RL models used for evaluation.

---

### Step 2 — Evaluate the Agents

```bash
python evaluate.py
```

This generates:

```text
results/comparison.png
results/summary.csv
```

The evaluation compares the behavior and performance of the **Safe RL** and **Plain RL** agents on test data.

---

### Step 3 — Streamlit Dashboard

Run:

```bash
streamlit run dashboard.py
```

The dashboard can be used to demonstrate the project during a **viva or project presentation**.

---

### Step 4 — Standalone HTML Dashboard

Generate the latest dashboard:

```bash
python frontend/build_dashboard.py
```

Then open:

```text
frontend/dashboard.html
```

The dashboard works directly in a browser and does not require a server or external CDN.

---

## 📊 Experimental Comparison

The main experiment compares:

### Safe RL

```text
RL Action
    ↓
Safety Shield
    ↓
Corrected Safe Action
    ↓
Battery
```

### Plain RL

```text
RL Action
    ↓
Battery
```

The comparison can be used to study:

* Grid electricity consumption
* Battery SOC behavior
* Unsafe actions
* Reward obtained by the agents
* Effect of safety constraints

---

## 🛡️ Why Use a Safety Shield?

A reinforcement learning agent may initially produce actions that are not physically valid.

For example:

```text
Battery SOC = 95%

RL Agent Action → Charge

Without Shield:
SOC → beyond allowed limit ❌

With Shield:
Action is reduced / clipped ✅
SOC remains within safe range
```

The important property of the safety layer is that it operates **independently of how well the RL policy has learned**.

This means battery safety can still be enforced even when the agent is poorly trained or produces unexpected actions.

---

## 📈 Experiment Parameters

Several parameters can be modified to study system behavior:

```text
grid_price_per_kwh
unsafe_penalty
battery capacity
maximum battery power
number of training timesteps
number of simulated days
```

Changing these parameters can be useful for a **sensitivity analysis** in a B.Tech report.

---

## 🧪 Synthetic Data

The project does not require downloading an external dataset.

`data_utils.py` generates synthetic profiles containing:

```text
solar_kw
demand_kw
```

This keeps the project lightweight and makes experimentation easier.

A real-world solar/load dataset can be integrated later using the same data format.

---

## 🔬 Original Research vs This Implementation

| Original Research Project         | This Implementation                       |
| --------------------------------- | ----------------------------------------- |
| OpenModelica + FMU simulation     | Pure Python simulation                    |
| Complex microgrid model           | Simplified battery + solar + demand model |
| OpenModelica Gym integration      | Custom Gymnasium environment              |
| Polytope / feasible-set safeguard | Rule-based Safety Shield                  |
| PPO reinforcement learning        | PPO reinforcement learning                |
| Optuna + MongoDB experiments      | CSV-based experiment results              |

The goal is not to reproduce the complete research system, but to provide a **simplified and understandable implementation of the core safe-RL concept**.

---

## 🚀 Future Improvements

Possible extensions include:

* Replace synthetic profiles with real solar/load datasets
* Add EV charging to the microgrid
* Support multiple batteries
* Add solar/load forecasting
* Compare PPO with SAC or DDPG
* Add more advanced safety constraints
* Introduce real-time energy pricing
* Improve the dashboard with additional experiment visualizations

---

## 🎓 Academic Context

This project is designed as a **beginner-friendly B.Tech implementation** of Safe Reinforcement Learning for energy management.

It demonstrates concepts from:

* Reinforcement Learning
* PPO
* Energy Storage Systems
* Microgrids
* Constraint Handling
* Safe AI
* Simulation

---

