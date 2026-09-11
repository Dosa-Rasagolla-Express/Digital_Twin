"""
twin_engine.py
===============
Real-Time Digital Twin as the Corridor Decision Engine (Novelty N1).

Implements Patent Draft Section 7.2 / Step 5 (the core inventive step) and
Claim 1(f)-(h):

  1. Candidate Schedule Generator — for the ambulance's route, enumerate N
     candidate corridor schedules, each varying per-junction offset and
     green-extension seconds (which approaches go green, phase order,
     offsets, green extensions, clearance intervals).

  2. Twin Simulator — fast-forward-simulate each candidate against the
     predicted network state (ETA bands from eta_predictor.py + live
     queue lengths), producing per-candidate metrics: ambulance delay,
     conflicting-traffic delay, stop count.

  3. Schedule Selector — discard candidates that violate hard constraints
     (minimum pedestrian walk phase, mandatory yellow + all-red) outright
     — never merely penalize them — then pick the argmin of the combined
     objective J = w1*ambulance_delay + w2*conflicting_delay + w3*stops
     among the survivors. This makes the result regulator-compliant by
     construction (patent Novelty N4), not just usually-compliant.

Unlike arbitration.py (which resolves *which* routes get priority) and
degradation.py (which tracks system *health*), this module is the piece
that actually decides *what the signal controllers should do* for a
given corridor — the "twin as decision engine" itself. It is deliberately
generic: it does not fast-forward a full microsimulation, but a closed-form
per-junction delay model, which is enough to make generate -> simulate ->
select a real, inspectable pipeline rather than a stub.
"""

import random
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from eta_predictor import ETABand, queue_clear_time_s


# ─────────────────────────────────────────────
# HARD CONSTRAINTS & OBJECTIVE WEIGHTS (patent §21 ranges, §7.2 Step 5)
# ─────────────────────────────────────────────
MIN_PEDESTRIAN_WALK_S  = 7.0     # within 3-60 s
MIN_YELLOW_S           = 2.0     # within 2-10 s
MIN_ALL_RED_S          = 2.0     # within 1-30 s
MAX_GREEN_EXTENSION_S  = 60.0    # kept modest for demo; patent allows up to 300 s

# Operator/policy-configurable objective weights (must sum to ~1.0, patent §7.2)
W_AMBULANCE    = 0.6
W_CONFLICTING  = 0.3
W_STOPS        = 0.1


# ─────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────

@dataclass
class CandidateSchedule:
    schedule_id: str
    junction_offsets_s: Dict[str, float]     # seconds before the corridor approach goes green
    green_extension_s: Dict[str, float]      # per-junction green-extension seconds
    yellow_s: float
    all_red_s: float
    pedestrian_walk_s: float


@dataclass
class ScheduleMetrics:
    schedule: CandidateSchedule
    ambulance_delay_s: float
    conflicting_delay_s: float
    stop_count: int
    objective_j: float
    feasible: bool
    infeasibility_reason: str = ""


# ─────────────────────────────────────────────
# 1. CANDIDATE SCHEDULE GENERATOR
# ─────────────────────────────────────────────

def generate_candidates(
    route: List[str],
    n: int = 12,
    seed: str = "",
) -> List[CandidateSchedule]:
    """
    Enumerate N candidate corridor schedules for the downstream junctions
    of `route` (patent Step 5, typical N: 2-100). Most candidates are
    schedule-compliant by construction; a few deliberately violate a hard
    constraint so the Schedule Selector has something real to reject.
    """
    rng = random.Random(seed or "-".join(route))
    downstream = route[1:] or route

    offset_choices    = [0.0, 2.0, 4.0, 6.0, 8.0]
    extension_choices = [15.0, 25.0, 35.0, 45.0]

    candidates = []
    for i in range(n):
        # ~1 in 6 candidates is intentionally non-compliant (e.g. shortened
        # pedestrian phase) to demonstrate the Selector's hard-constraint gate.
        noncompliant = rng.random() < (1.0 / 6.0)

        candidates.append(CandidateSchedule(
            schedule_id=f"S{i + 1}",
            junction_offsets_s={j: rng.choice(offset_choices) for j in downstream},
            green_extension_s={j: rng.choice(extension_choices) for j in downstream},
            yellow_s=MIN_YELLOW_S if not noncompliant else rng.choice([0.5, 1.0]),
            all_red_s=rng.choice([2.0, 3.0, 4.0]) if not noncompliant else rng.choice([0.0, 1.0]),
            pedestrian_walk_s=rng.choice([7.0, 10.0, 12.0]) if not noncompliant else rng.choice([2.0, 4.0]),
        ))
    return candidates


# ─────────────────────────────────────────────
# 2. TWIN SIMULATOR
# ─────────────────────────────────────────────

def _feasible(candidate: CandidateSchedule) -> Tuple[bool, str]:
    """Hard constraints (patent Novelty N4) — violating any one makes the
    whole candidate infeasible, regardless of how good its objective score
    would otherwise be."""
    if candidate.pedestrian_walk_s < MIN_PEDESTRIAN_WALK_S:
        return False, (f"pedestrian walk {candidate.pedestrian_walk_s}s "
                        f"< minimum {MIN_PEDESTRIAN_WALK_S}s")
    if candidate.yellow_s < MIN_YELLOW_S:
        return False, f"yellow {candidate.yellow_s}s < minimum {MIN_YELLOW_S}s"
    if candidate.all_red_s < MIN_ALL_RED_S:
        return False, f"all-red {candidate.all_red_s}s < minimum {MIN_ALL_RED_S}s"
    for j, ext in candidate.green_extension_s.items():
        if ext > MAX_GREEN_EXTENSION_S:
            return False, f"{j} green-extension {ext}s exceeds cap {MAX_GREEN_EXTENSION_S}s"
    return True, ""


def simulate_candidate(
    candidate: CandidateSchedule,
    route: List[str],
    eta_bands: Dict[str, ETABand],
    queue_lengths: Dict[str, int],
) -> ScheduleMetrics:
    """
    Fast-forward-simulate one candidate against the predicted network
    state (patent Step 5): at each downstream junction, the ambulance is
    delayed by whichever is later — the corridor actually turning green
    (offset + transition) or the queue ahead of it clearing; conflicting
    traffic is held for the corridor's green-extension + all-red window.
    """
    feasible, reason = _feasible(candidate)

    ambulance_delay = 0.0
    conflicting_delay = 0.0
    stops = 0

    for junction in route[1:]:
        offset = candidate.junction_offsets_s.get(junction, 0.0)
        extension = candidate.green_extension_s.get(junction, 0.0)
        band = eta_bands.get(junction)
        arrival = band.mean_s if band else 0.0
        queue = queue_lengths.get(junction, 0)
        clear_time = queue_clear_time_s(queue)

        ready_time = max(offset + candidate.yellow_s + candidate.all_red_s, clear_time)
        delay = max(0.0, ready_time - arrival)
        ambulance_delay += delay
        if delay > 1.0:
            stops += 1

        conflicting_delay += extension * 0.5 + candidate.all_red_s

    objective = (
        W_AMBULANCE * ambulance_delay
        + W_CONFLICTING * conflicting_delay
        + W_STOPS * stops * 10.0
    )
    if not feasible:
        objective = float("inf")

    return ScheduleMetrics(
        schedule=candidate,
        ambulance_delay_s=round(ambulance_delay, 1),
        conflicting_delay_s=round(conflicting_delay, 1),
        stop_count=stops,
        objective_j=objective if not feasible else round(objective, 2),
        feasible=feasible,
        infeasibility_reason=reason,
    )


# ─────────────────────────────────────────────
# 3. SCHEDULE SELECTOR
# ─────────────────────────────────────────────

def select_schedule(
    route: List[str],
    eta_bands: Dict[str, ETABand],
    queue_lengths: Dict[str, int],
    n_candidates: int = 12,
    seed: str = "",
) -> Tuple[ScheduleMetrics, List[ScheduleMetrics]]:
    """
    Generate -> simulate -> select (patent Step 5). Returns
    (winner, all_candidates_ranked_by_objective).

    If every candidate is infeasible (a fail-safe edge case), the "winner"
    returned is simply the best-ranked infeasible one, flagged
    `feasible=False` — the caller should treat this as a signal to fall
    back to CONSERVATIVE_HOLD (patent Step 8 / degradation.py) rather than
    execute it.
    """
    candidates = generate_candidates(route, n=n_candidates, seed=seed)
    scored = [simulate_candidate(c, route, eta_bands, queue_lengths) for c in candidates]
    ranked = sorted(scored, key=lambda s: s.objective_j)

    feasible_scored = [s for s in scored if s.feasible]
    winner = min(feasible_scored, key=lambda s: s.objective_j) if feasible_scored else ranked[0]
    return winner, ranked


# ─────────────────────────────────────────────
# STANDALONE DEMO
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # emoji-safe on Windows consoles (cp1252)
    except AttributeError:
        pass

    from eta_predictor import eta_bands_for_route

    route = ["West Junction", "Main Junction", "North Junction"]
    bands = eta_bands_for_route(route)
    queues = {"Main Junction": 12, "North Junction": 6}

    winner, ranked = select_schedule(route, bands, queues, n_candidates=12, seed="demo")

    print(f"Route: {' -> '.join(route)}")
    print(f"Evaluated {len(ranked)} candidates ({sum(s.feasible for s in ranked)} feasible)\n")

    print("Top 5 candidates:")
    for m in ranked[:5]:
        flag = "OK " if m.feasible else "REJ"
        j_str = f"{m.objective_j:7.2f}" if m.feasible else "   inf "
        print(f"  [{flag}] {m.schedule.schedule_id:4s} J={j_str}  "
              f"amb_delay={m.ambulance_delay_s:5.1f}s  cross_delay={m.conflicting_delay_s:5.1f}s  "
              f"stops={m.stop_count}"
              + (f"  -- {m.infeasibility_reason}" if not m.feasible else ""))

    print(f"\nSelected schedule: {winner.schedule.schedule_id} "
          f"(feasible={winner.feasible}, J={winner.objective_j})")
