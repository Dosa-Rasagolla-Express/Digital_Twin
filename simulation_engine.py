"""
simulation_engine.py
====================
Traffic scenario simulation engine.

Generates synthetic traffic data for different city scenarios:
  - Normal
  - Rain Mode
  - Accident Mode
  - Road Block
  - Festival Traffic
  - School Rush Hour

Each scenario modifies vehicle counts, speeds, and congestion
levels for all five simulated junctions.
"""

import random
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime


# ─────────────────────────────────────────────
# JUNCTION DEFINITIONS
# ─────────────────────────────────────────────

JUNCTIONS = {
    "Main Junction":  {"lat": 12.9716, "lon": 77.5946, "base_traffic": 15},
    "North Junction": {"lat": 12.9740, "lon": 77.5946, "base_traffic": 10},
    "South Junction": {"lat": 12.9685, "lon": 77.5946, "base_traffic": 18},
    "East Junction":  {"lat": 12.9718, "lon": 77.5980, "base_traffic": 8},
    "West Junction":  {"lat": 12.9700, "lon": 77.5910, "base_traffic": 12},
}


# ─────────────────────────────────────────────
# SCENARIO PROFILES
# ─────────────────────────────────────────────

SCENARIO_PROFILES = {
    "Normal": {
        "traffic_mult": 1.0,
        "speed_mult":   1.0,
        "ambulance_prob": 0.03,
        "noise":         3,
        "description":   "Normal city traffic — baseline conditions.",
        "icon":          "🏙️",
    },
    "Rain Mode": {
        "traffic_mult": 1.35,
        "speed_mult":   0.7,
        "ambulance_prob": 0.05,
        "noise":         5,
        "description":   "Heavy rain slows vehicles and increases density.",
        "icon":          "🌧️",
    },
    "Accident Mode": {
        "traffic_mult": 1.8,
        "speed_mult":   0.4,
        "ambulance_prob": 0.25,
        "noise":         8,
        "description":   "Accident causes major slow-down and emergency response.",
        "icon":          "🚨",
    },
    "Road Block": {
        "traffic_mult": 2.2,
        "speed_mult":   0.2,
        "ambulance_prob": 0.10,
        "noise":         6,
        "description":   "Road closed — traffic diverted through alternative routes.",
        "icon":          "🚧",
    },
    "Festival Traffic": {
        "traffic_mult": 2.5,
        "speed_mult":   0.5,
        "ambulance_prob": 0.04,
        "noise":         10,
        "description":   "City festival — peak crowd movement expected.",
        "icon":          "🎉",
    },
    "School Rush Hour": {
        "traffic_mult": 1.6,
        "speed_mult":   0.65,
        "ambulance_prob": 0.03,
        "noise":         4,
        "description":   "School opening/closing — elevated pedestrian and vehicle activity.",
        "icon":          "🏫",
    },
}


# ─────────────────────────────────────────────
# DATA CLASSES
# ─────────────────────────────────────────────

@dataclass
class JunctionState:
    name: str
    lat: float
    lon: float
    vehicle_count: int
    avg_speed: float
    congestion: str
    ambulance: bool
    signal: str
    color: List[int] = field(default_factory=lambda: [0, 220, 90])
    radius: int = 300


@dataclass
class SimulationFrame:
    scenario: str
    timestamp: str
    junctions: Dict[str, JunctionState]
    total_vehicles: int
    global_ambulance: bool
    alerts: List[str] = field(default_factory=list)


# ─────────────────────────────────────────────
# SIMULATION ENGINE
# ─────────────────────────────────────────────

class SimulationEngine:
    """
    Generates traffic state for all junctions given a scenario name.
    """

    def __init__(self):
        self._tick = 0           # internal frame counter for wave simulation

    def _congestion_label(self, count: int) -> str:
        if count < 10:
            return "LOW"
        elif count < 20:
            return "MEDIUM"
        return "HIGH"

    def _congestion_color(self, level: str) -> List[int]:
        return {
            "LOW":       [0, 220, 90],
            "MEDIUM":    [255, 200, 0],
            "HIGH":      [255, 60, 60],
            "EMERGENCY": [180, 0, 255],
        }.get(level, [128, 128, 128])

    def _signal(self, count: int, ambulance: bool) -> str:
        if ambulance:
            return "GREEN (OVERRIDE)"
        if count < 10:
            return "GREEN"
        elif count < 20:
            return "YELLOW"
        return "RED"

    # --------------------------------------------------
    def generate_frame(
        self,
        scenario: str = "Normal",
        base_count_override: Optional[int] = None,
        manual_override: bool = False,
        emergency_mode: bool = False,
    ) -> SimulationFrame:
        """
        Produce one simulation frame for the given scenario.

        Parameters
        ----------
        scenario             : scenario name key
        base_count_override  : if provided, use as the base vehicle count
        manual_override      : manual signal override flag
        emergency_mode       : force global ambulance activation
        """
        self._tick += 1
        profile = SCENARIO_PROFILES.get(scenario, SCENARIO_PROFILES["Normal"])

        mult       = profile["traffic_mult"]
        speed_mult = profile["speed_mult"]
        amb_prob   = profile["ambulance_prob"]
        noise      = profile["noise"]

        global_ambulance = emergency_mode or (random.random() < amb_prob)
        junctions_out: Dict[str, JunctionState] = {}
        alerts: List[str] = []
        total_vehicles = 0

        for jname, jinfo in JUNCTIONS.items():
            base = base_count_override if base_count_override is not None \
                   else jinfo["base_traffic"]

            # Add a sinusoidal traffic wave + random noise
            wave = math.sin(self._tick * 0.1 + hash(jname) % 10) * 3
            vc   = max(0, int(base * mult + wave + random.randint(-noise, noise)))

            # Speed inversely proportional to vehicle count
            spd = max(5.0, (60 - vc * 0.8) * speed_mult + random.uniform(-3, 3))

            amb  = global_ambulance and (jname == "Main Junction")
            cong = self._congestion_label(vc)
            if amb:
                cong = "HIGH"

            sig   = self._signal(vc, amb or (manual_override and emergency_mode))
            color = self._congestion_color("EMERGENCY" if amb else cong)

            # Bubble radius scales with vehicle count
            radius = 200 + vc * 15

            junctions_out[jname] = JunctionState(
                name=jname,
                lat=jinfo["lat"],
                lon=jinfo["lon"],
                vehicle_count=vc,
                avg_speed=round(spd, 1),
                congestion=cong,
                ambulance=amb,
                signal=sig,
                color=color,
                radius=radius,
            )
            total_vehicles += vc

        # ── Generate alerts ──────────────────────────────
        for jname, js in junctions_out.items():
            if js.ambulance:
                alerts.append(f"🚑 AMBULANCE detected at {jname} — Emergency corridor active!")
            if js.congestion == "HIGH":
                alerts.append(f"🔴 Heavy congestion at {jname} ({js.vehicle_count} vehicles)")
        if scenario == "Accident Mode":
            alerts.append("🚨 ACCIDENT ZONE — Emergency services dispatched")
        if scenario == "Road Block":
            alerts.append("🚧 ROAD BLOCK active — Traffic rerouted")

        return SimulationFrame(
            scenario=scenario,
            timestamp=datetime.now().strftime("%H:%M:%S"),
            junctions=junctions_out,
            total_vehicles=total_vehicles,
            global_ambulance=global_ambulance,
            alerts=alerts,
        )

    # --------------------------------------------------
    def junction_to_pydeck_df(self, frame: SimulationFrame):
        """
        Convert a SimulationFrame into a pandas DataFrame
        suitable for PyDeck ScatterplotLayer / HeatmapLayer.
        """
        import pandas as pd
        rows = []
        for jname, js in frame.junctions.items():
            rows.append({
                "junction":      jname,
                "lat":           js.lat,
                "lon":           js.lon,
                "vehicles":      js.vehicle_count,
                "avg_speed":     js.avg_speed,
                "congestion":    js.congestion,
                "signal":        js.signal,
                "ambulance":     int(js.ambulance),
                "color":         js.color,
                "radius":        js.radius,
            })
        return pd.DataFrame(rows)

    # --------------------------------------------------
    def generate_history(
        self,
        scenario: str = "Normal",
        n_steps: int = 30,
    ):
        """
        Generate n_steps of historical simulation data for charts.
        Returns a pandas DataFrame with columns [step, vehicles, speed, congestion].
        """
        import pandas as pd
        rows = []
        for step in range(n_steps):
            frame = self.generate_frame(scenario)
            main  = frame.junctions.get("Main Junction")
            if main:
                rows.append({
                    "step":       step,
                    "vehicles":   main.vehicle_count,
                    "speed":      main.avg_speed,
                    "congestion": main.congestion,
                })
        return pd.DataFrame(rows)


# ─────────────────────────────────────────────
# MODULE-LEVEL INSTANCE
# ─────────────────────────────────────────────
_engine = SimulationEngine()


def get_simulation_frame(
    scenario: str = "Normal",
    base_count: Optional[int] = None,
    emergency_mode: bool = False,
) -> SimulationFrame:
    """Convenience wrapper — generate one simulation frame."""
    return _engine.generate_frame(
        scenario=scenario,
        base_count_override=base_count,
        emergency_mode=emergency_mode,
    )


def get_scenario_names() -> List[str]:
    """Return list of available scenario names."""
    return list(SCENARIO_PROFILES.keys())


def get_scenario_info(scenario: str) -> dict:
    """Return the profile dict for a given scenario."""
    return SCENARIO_PROFILES.get(scenario, SCENARIO_PROFILES["Normal"])
