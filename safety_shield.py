"""
safety_shield.py
-----------------
This is the "safe" part of Safe RL. It plays the same role as the
polytope/feasible-set safeguard in the original research repo, but with plain
bounds-checking instead of control-theory math.

The idea: whatever action the RL agent proposes, we PREDICT what battery
state-of-charge (SOC) it would lead to. If that predicted SOC would leave the
safe operating window [soc_min, soc_max], we clip the action to the largest
value that keeps SOC exactly inside the window. The agent's action is only
ever modified, never executed as-is if it's unsafe.
"""

from dataclasses import dataclass


@dataclass
class BatteryParams:
    capacity_kwh: float = 10.0     # total usable battery capacity
    max_power_kw: float = 3.0      # max charge/discharge power
    efficiency: float = 0.95       # round-trip efficiency
    soc_min: float = 0.10          # never discharge below 10%
    soc_max: float = 0.95          # never charge above 95%
    dt_hours: float = 1.0          # length of one simulation step


def predict_next_soc(soc: float, action: float, params: BatteryParams) -> float:
    """
    action is in [-1, 1]: -1 = full discharge, +1 = full charge.
    Returns the SOC (0-1) that would result from applying `action` for one step.
    """
    power_kw = action * params.max_power_kw
    if power_kw >= 0:  # charging: some energy is lost to inefficiency
        delta_soc = (power_kw * params.efficiency * params.dt_hours) / params.capacity_kwh
    else:               # discharging: need MORE stored energy than delivered
        delta_soc = (power_kw / params.efficiency * params.dt_hours) / params.capacity_kwh
    return soc + delta_soc


def shield_action(soc: float, action: float, params: BatteryParams) -> tuple[float, bool]:
    """
    Core safety function.

    Returns:
        safe_action: the (possibly modified) action that is guaranteed to keep
                      SOC inside [soc_min, soc_max]
        was_clipped:  True if the agent's original action had to be corrected
    """
    predicted_soc = predict_next_soc(soc, action, params)

    if params.soc_min <= predicted_soc <= params.soc_max:
        return action, False  # already safe, nothing to do

    # Unsafe -> find the boundary action that lands exactly on soc_min/soc_max.
    target_soc = params.soc_max if predicted_soc > params.soc_max else params.soc_min
    delta_soc_needed = target_soc - soc

    if delta_soc_needed >= 0:
        power_kw = (delta_soc_needed * params.capacity_kwh) / (params.efficiency * params.dt_hours)
    else:
        power_kw = (delta_soc_needed * params.capacity_kwh * params.efficiency) / params.dt_hours

    safe_action = power_kw / params.max_power_kw
    safe_action = max(-1.0, min(1.0, safe_action))  # keep inside actuator limits too
    return safe_action, True


if __name__ == "__main__":
    # quick manual sanity check
    p = BatteryParams()
    print("Battery almost full (soc=0.94), agent wants to charge hard (action=1.0):")
    print(" ->", shield_action(0.94, 1.0, p))

    print("Battery almost empty (soc=0.11), agent wants to discharge hard (action=-1.0):")
    print(" ->", shield_action(0.11, -1.0, p))

    print("Battery mid (soc=0.5), agent wants action=0.3 (should be untouched):")
    print(" ->", shield_action(0.5, 0.3, p))
