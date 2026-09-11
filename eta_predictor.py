"""
eta_predictor.py
=================
Probabilistic Route Prediction & Uncertainty-Aware ETA-Band Estimator
(Novelty N3).

Implements Patent Draft Section 7.2 / Step 4 / Step 6 / Claim 1(d),(h):

  - Route predictor: scores candidate downstream paths on the network
    graph from the ambulance's current junction, returning a route
    probability distribution over the top-K hypotheses (shorter /
    higher-scoring continuations are more probable — a stand-in for the
    trained model described in the patent, which would additionally
    condition on heading, historical route priors, and live congestion).

  - ETA-band estimator: for each downstream junction on a route, returns
    an arrival-time distribution (mean + standard deviation) whose
    uncertainty compounds with hop count, plus an upper-confidence-bound
    (UCB) at a configurable confidence level — never a bare point
    estimate (that is precisely the Denso-type / MLEVP-type limitation
    this module is designed to avoid, per patent Novelty N3).

  - Trigger rule (Step 6): corridor preparation at junction i fires when

        UCB(ETA_i) - t_clear(queue_i) <= margin

    ensuring a prepared corridor is ready even for a slower-than-expected
    arrival, while avoiding premature occupation of the intersection.
"""

import math
from dataclasses import dataclass
from typing import Dict, List

import networkx as nx
from scipy.stats import norm

from network_model import JUNCTION_EDGES, junction_distance_m


# ─────────────────────────────────────────────
# CONFIGURATION (patent §21 broad workable ranges)
# ─────────────────────────────────────────────
DEFAULT_SPEED_KMH     = 42.0     # assumed emergency-vehicle cruising speed
SPEED_STD_FRACTION    = 0.18     # std as a fraction of cumulative mean travel time
CONFIDENCE_LEVEL      = 0.90     # within 50-99.99%
TRIGGER_MARGIN_S      = 8.0      # within 0-60 s, per-intersection tunable
SATURATION_HEADWAY_S  = 2.2      # seconds of green consumed clearing one queued vehicle
ROUTE_HORIZON_HOPS    = 3        # within 1-20 downstream intersections


# ─────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────

@dataclass
class RouteHypothesis:
    path: List[str]
    probability: float


@dataclass
class ETABand:
    junction: str
    hop_index: int
    mean_s: float
    std_s: float
    ucb_s: float
    confidence_level: float


@dataclass
class TriggerDecision:
    junction: str
    ucb_eta_s: float
    queue_clear_s: float
    margin_s: float
    should_trigger: bool
    reason: str


# ─────────────────────────────────────────────
# ROUTE PREDICTION (Step 4)
# ─────────────────────────────────────────────

def predict_routes(
    origin: str,
    k: int = 3,
    max_hops: int = ROUTE_HORIZON_HOPS,
) -> List[RouteHypothesis]:
    """
    Score candidate downstream paths from `origin` on the network graph
    and return the top-K as a normalized probability distribution.

    Shorter / more-direct continuations score higher, which — absent a
    trained heading/history model in this demo — is a reasonable proxy
    for "most probable path" on a small road network.
    """
    G = nx.Graph()
    G.add_edges_from(JUNCTION_EDGES)
    if origin not in G:
        return []

    candidate_paths = []
    for target in G.nodes:
        if target == origin:
            continue
        for path in nx.all_simple_paths(G, origin, target, cutoff=max_hops):
            candidate_paths.append(path)

    if not candidate_paths:
        return []

    scored = [(p, 1.0 / len(p)) for p in candidate_paths]
    scored.sort(key=lambda x: -x[1])
    top = scored[:k]
    total = sum(s for _, s in top) or 1.0
    return [
        RouteHypothesis(path=p, probability=round(s / total, 3))
        for p, s in top
    ]


# ─────────────────────────────────────────────
# ETA-BAND ESTIMATION (Step 4 / Claim 1(d))
# ─────────────────────────────────────────────

def eta_bands_for_route(
    route: List[str],
    speed_kmh: float = DEFAULT_SPEED_KMH,
    confidence_level: float = CONFIDENCE_LEVEL,
) -> Dict[str, ETABand]:
    """
    For each downstream junction on `route` (excluding the origin), return
    an ETABand: mean/std arrival time from now and an upper-confidence-bound
    at `confidence_level`. Uncertainty compounds with hop count, reflecting
    that farther-downstream arrivals are harder to predict precisely.
    """
    z = norm.ppf(confidence_level)
    bands: Dict[str, ETABand] = {}
    cumulative_mean = 0.0
    speed_ms = speed_kmh * 1000 / 3600

    for hop_index, (a, b) in enumerate(zip(route, route[1:]), start=1):
        dist_m = junction_distance_m(a, b)
        travel_s = dist_m / speed_ms
        cumulative_mean += travel_s
        std = cumulative_mean * SPEED_STD_FRACTION * math.sqrt(hop_index)
        ucb = cumulative_mean + z * std

        bands[b] = ETABand(
            junction=b,
            hop_index=hop_index,
            mean_s=round(cumulative_mean, 1),
            std_s=round(std, 1),
            ucb_s=round(ucb, 1),
            confidence_level=confidence_level,
        )

    return bands


# ─────────────────────────────────────────────
# UNCERTAINTY-AWARE TRIGGER RULE (Step 6)
# ─────────────────────────────────────────────

def queue_clear_time_s(queue_length: int) -> float:
    """t_clear(queue): seconds of green needed to clear the queue ahead
    of the emergency-vehicle approach at one junction."""
    return max(0.0, queue_length) * SATURATION_HEADWAY_S


def trigger_decision(
    junction: str,
    eta_band: ETABand,
    queue_length: int,
    margin_s: float = TRIGGER_MARGIN_S,
) -> TriggerDecision:
    """
    Patent Step 6: fire corridor preparation at `junction` when
        UCB(ETA) - t_clear(queue) <= margin
    """
    t_clear = queue_clear_time_s(queue_length)
    slack = eta_band.ucb_s - t_clear
    should_trigger = slack <= margin_s

    if should_trigger:
        reason = (
            f"UCB({eta_band.ucb_s:.1f}s) - t_clear({t_clear:.1f}s) = "
            f"{slack:.1f}s <= margin({margin_s:.1f}s) — prepare corridor now."
        )
    else:
        reason = (
            f"UCB({eta_band.ucb_s:.1f}s) - t_clear({t_clear:.1f}s) = "
            f"{slack:.1f}s > margin({margin_s:.1f}s) — not yet; queue will "
            f"clear well before the worst-case arrival."
        )

    return TriggerDecision(
        junction=junction,
        ucb_eta_s=eta_band.ucb_s,
        queue_clear_s=round(t_clear, 1),
        margin_s=margin_s,
        should_trigger=should_trigger,
        reason=reason,
    )


def trigger_decisions_for_route(
    route: List[str],
    eta_bands: Dict[str, ETABand],
    queue_lengths: Dict[str, int],
    margin_s: float = TRIGGER_MARGIN_S,
) -> List[TriggerDecision]:
    """Convenience wrapper: trigger_decision() for every downstream junction on the route."""
    decisions = []
    for junction in route[1:]:
        band = eta_bands.get(junction)
        if not band:
            continue
        decisions.append(
            trigger_decision(junction, band, queue_lengths.get(junction, 0), margin_s)
        )
    return decisions


# ─────────────────────────────────────────────
# STANDALONE DEMO
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # emoji-safe on Windows consoles (cp1252)
    except AttributeError:
        pass

    origin = "West Junction"
    hypotheses = predict_routes(origin, k=3)
    print(f"Route hypotheses from {origin}:")
    for h in hypotheses:
        print(f"  p={h.probability:.2f}  {' -> '.join(h.path)}")

    best_route = hypotheses[0].path if hypotheses else [origin]
    print(f"\nETA bands for top route: {' -> '.join(best_route)}")
    bands = eta_bands_for_route(best_route)
    for j, band in bands.items():
        print(f"  {j:16s} mean={band.mean_s:6.1f}s  std={band.std_s:5.1f}s  "
              f"UCB90={band.ucb_s:6.1f}s")

    demo_queues = {j: q for j, q in zip(best_route[1:], [4, 9])}
    print("\nTrigger decisions:")
    for d in trigger_decisions_for_route(best_route, bands, demo_queues):
        flag = "TRIGGER" if d.should_trigger else "hold"
        print(f"  [{flag:7s}] {d.junction:16s} {d.reason}")
