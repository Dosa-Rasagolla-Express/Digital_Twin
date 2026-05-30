"""
signal_optimizer.py
===================
Smart traffic signal timing optimizer.

Rules
-----
Traffic < 10   →  20 s green
Traffic 10-20  →  40 s green
Traffic 20-30  →  60 s green
Traffic > 30   →  90 s green

Emergency override: ambulance detected → 90 s green + corridor activation.
"""

from dataclasses import dataclass, field
from typing import Dict, List


# ─────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────

@dataclass
class SignalRecommendation:
    junction: str
    vehicle_count: int
    green_time: int          # seconds
    red_time: int            # seconds
    yellow_time: int = 5     # always 5 s
    emergency_override: bool = False
    priority_route: List[str] = field(default_factory=list)
    reason: str = ""


# ─────────────────────────────────────────────
# CORE OPTIMIZER FUNCTIONS
# ─────────────────────────────────────────────

def calculate_green_time(vehicle_count: int, ambulance: bool = False) -> int:
    """
    Return recommended green-signal duration in seconds.

    Parameters
    ----------
    vehicle_count : current vehicle count at the junction
    ambulance     : True if an emergency vehicle is detected
    """
    if ambulance:
        return 90  # Always max green for emergency corridor

    if vehicle_count < 10:
        return 20
    elif vehicle_count < 20:
        return 40
    elif vehicle_count < 30:
        return 60
    else:
        return 90


def calculate_red_time(green_time: int, yellow_time: int = 5) -> int:
    """
    Compute complementary red time so the full cycle stays ≤ 120 s.
    Formula: red = 120 − green − yellow (minimum 20 s).
    """
    red = 120 - green_time - yellow_time
    return max(20, red)


def get_signal_recommendation(
    junction: str,
    vehicle_count: int,
    ambulance: bool = False,
) -> SignalRecommendation:
    """
    Return a full SignalRecommendation for a single junction.
    """
    green  = calculate_green_time(vehicle_count, ambulance)
    red    = calculate_red_time(green)

    if ambulance:
        reason = "🚑 Emergency override — ambulance corridor activated"
        priority = [
            f"{junction} → Main Junction → Hospital Route",
            "All crossing signals set to RED",
        ]
        return SignalRecommendation(
            junction=junction,
            vehicle_count=vehicle_count,
            green_time=green,
            red_time=red,
            emergency_override=True,
            priority_route=priority,
            reason=reason,
        )

    if vehicle_count < 10:
        reason = "Light traffic — minimal green time"
    elif vehicle_count < 20:
        reason = "Moderate traffic — standard green time"
    elif vehicle_count < 30:
        reason = "Heavy traffic — extended green time"
    else:
        reason = "Severe congestion — maximum green time"

    return SignalRecommendation(
        junction=junction,
        vehicle_count=vehicle_count,
        green_time=green,
        red_time=red,
        reason=reason,
    )


def optimize_all_junctions(
    junction_data: Dict[str, dict],
    global_ambulance: bool = False,
) -> Dict[str, SignalRecommendation]:
    """
    Optimize signal timings for all junctions simultaneously.

    Parameters
    ----------
    junction_data : {junction_name: {"vehicle_count": int, "ambulance": bool}}
    global_ambulance : True if any junction has an ambulance

    Returns
    -------
    dict of {junction_name: SignalRecommendation}
    """
    recommendations = {}

    for jname, info in junction_data.items():
        if isinstance(info, int):
            vc = info
            amb = global_ambulance
        else:
            vc = info.get("vehicle_count", 0)
            amb = info.get("ambulance", False) or global_ambulance

        recommendations[jname] = get_signal_recommendation(
            jname, vc, amb
        )

    return recommendations


# ─────────────────────────────────────────────
# CONGESTION CLASSIFIER
# ─────────────────────────────────────────────

def classify_congestion(vehicle_count: int) -> str:
    """Return LOW / MEDIUM / HIGH based on vehicle count."""
    if vehicle_count < 10:
        return "LOW"
    elif vehicle_count < 20:
        return "MEDIUM"
    else:
        return "HIGH"


def congestion_color_rgb(level: str) -> list:
    """Return [R, G, B] for PyDeck coloring."""
    return {
        "LOW":    [0, 220, 90],
        "MEDIUM": [255, 200, 0],
        "HIGH":   [255, 60, 60],
    }.get(level, [128, 128, 128])


def smart_green_time(vehicle_count: int, avg_speed: float, ambulance: bool) -> tuple:
    """
    Legacy helper used by main.py.
    Returns (green_time_seconds, congestion_level).
    """
    if ambulance:
        return 90, "EMERGENCY"

    green = calculate_green_time(vehicle_count)
    cong  = classify_congestion(vehicle_count)
    return green, cong
