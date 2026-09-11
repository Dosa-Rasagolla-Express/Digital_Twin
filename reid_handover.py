"""
reid_handover.py
=================
GPS-Free Cross-Camera Vehicle Identity Handover (Novelty N2).

Implements Patent Draft Section 7.1 / Step 3 / Claim 2:

    As the ambulance exits camera k's field of view, an appearance
    embedding vector is extracted. At an adjacent camera k+1, candidate
    entry detections are matched against it. A match is confirmed only
    if ALL of the following gates pass:

      (a) embedding distance < threshold (appearance similarity),
      (b) exit bearing at camera k is compatible with the expected
          entry heading into camera k+1 (geometric plausibility), and
      (c) elapsed travel time implies a feasible speed between the two
          camera sites (temporal plausibility).

    On confirmation, the same unique corridor ID continues into camera
    k+1. If no match is confirmed within HANDOVER_TIMEOUT_S, the
    corridor is reported lost — the caller (dashboard/degradation.py)
    should then fall back to CONSERVATIVE_HOLD (patent Step 8) until
    reacquisition.

This module has no real camera/CV backend — embeddings are simulated as
fixed-length unit vectors with small per-hop noise, standing in for a
real ReID network's near-stable output across views. The matching/gating
logic itself (the actual inventive step) is real and exercised end-to-end.
"""

import math
import random
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from network_model import JUNCTION_EDGES, junction_distance_m, junction_bearing_deg


# ─────────────────────────────────────────────
# CONFIGURATION (patent §21 broad workable ranges)
# ─────────────────────────────────────────────
EMBEDDING_DISTANCE_THRESHOLD = 0.45     # within 0.2-0.9 (normalized)
MIN_FEASIBLE_SPEED_KMH       = 5.0      # within 5-150 km/h
MAX_FEASIBLE_SPEED_KMH       = 150.0
HANDOVER_TIMEOUT_S           = 20.0     # within 2-120 s
BEARING_TOLERANCE_DEG        = 60.0     # exit-bearing / entry-gate compatibility
EMBEDDING_DIM                = 32

# Camera adjacency reuses the junction topology: one road-facing camera
# per junction, adjacency = network edges.
CAMERA_ADJACENCY: Dict[str, List[str]] = {}
for _a, _b in JUNCTION_EDGES:
    CAMERA_ADJACENCY.setdefault(_a, []).append(_b)
    CAMERA_ADJACENCY.setdefault(_b, []).append(_a)


# ─────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────

@dataclass
class AppearanceEmbedding:
    """Simulated ReID embedding vector for one detection."""
    vector: Tuple[float, ...]

    @staticmethod
    def sample(seed_id: str, hop_noise: float = 0.08) -> "AppearanceEmbedding":
        """Deterministic-per-corridor embedding with small per-hop noise —
        stands in for a real appearance-ReID network's near-stable output
        across camera views of the same physical vehicle."""
        rng = random.Random(seed_id)
        base = [rng.uniform(-1, 1) for _ in range(EMBEDDING_DIM)]
        noisy = [v + random.uniform(-hop_noise, hop_noise) for v in base]
        norm = math.sqrt(sum(v * v for v in noisy)) or 1.0
        return AppearanceEmbedding(tuple(v / norm for v in noisy))

    def distance(self, other: "AppearanceEmbedding") -> float:
        """Normalized Euclidean distance between two unit-ish vectors."""
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(self.vector, other.vector)))


@dataclass
class CorridorTrack:
    """State of one ambulance's identity as it moves camera-to-camera."""
    corridor_id: str
    current_camera: str
    embedding: AppearanceEmbedding
    exit_time: Optional[datetime] = None
    exit_bearing: Optional[float] = None
    confirmed: bool = True
    hops: List["HandoverEvent"] = field(default_factory=list)


@dataclass
class HandoverEvent:
    corridor_id: str
    from_camera: str
    to_camera: str
    timestamp: datetime
    embedding_distance: float
    travel_time_s: float
    implied_speed_kmh: float
    bearing_delta_deg: float
    matched: bool
    reason: str


# ─────────────────────────────────────────────
# HANDOVER MANAGER
# ─────────────────────────────────────────────

class ReIDHandoverManager:
    """
    Tracks active corridor identities and performs gated cross-camera
    re-identification as each ambulance exits one camera's field of view
    and enters an adjacent one (patent Step 3 / Claim 2).
    """

    def __init__(self):
        self.tracks: Dict[str, CorridorTrack] = {}
        self.lost_since: Dict[str, datetime] = {}

    def acquire(self, corridor_id: str, camera: str, when: Optional[datetime] = None) -> CorridorTrack:
        """Mint a new corridor identity at first detection (patent Step 2)."""
        track = CorridorTrack(
            corridor_id=corridor_id,
            current_camera=camera,
            embedding=AppearanceEmbedding.sample(corridor_id),
        )
        self.tracks[corridor_id] = track
        self.lost_since.pop(corridor_id, None)
        return track

    def exit_camera(self, corridor_id: str, bearing_deg: float, when: Optional[datetime] = None) -> None:
        """Record that the ambulance has exited its current camera's field of view."""
        track = self.tracks.get(corridor_id)
        if not track:
            return
        track.exit_time = when or datetime.now()
        track.exit_bearing = bearing_deg
        track.confirmed = False

    def attempt_handover(
        self,
        corridor_id: str,
        candidate_camera: str,
        entry_embedding: Optional[AppearanceEmbedding] = None,
        when: Optional[datetime] = None,
    ) -> HandoverEvent:
        """
        Attempt to confirm the same identity at an adjacent camera
        (patent Claim 2): gate on embedding distance, exit-bearing/entry-gate
        compatibility, and travel-time feasibility.
        """
        when = when or datetime.now()
        track = self.tracks.get(corridor_id)
        if not track or track.exit_time is None:
            return HandoverEvent(
                corridor_id, "?", candidate_camera, when, 1.0, 0.0, 0.0, 0.0,
                matched=False, reason="No exit event recorded for this corridor.",
            )

        from_camera = track.current_camera
        entry_embedding = entry_embedding or AppearanceEmbedding.sample(corridor_id)
        emb_distance = track.embedding.distance(entry_embedding)

        travel_time_s = max(0.001, (when - track.exit_time).total_seconds())
        distance_m = junction_distance_m(from_camera, candidate_camera)
        implied_speed_kmh = (distance_m / travel_time_s) * 3.6

        expected_bearing = junction_bearing_deg(from_camera, candidate_camera)
        bearing_delta = abs(((track.exit_bearing - expected_bearing) + 180) % 360 - 180)

        adjacency_ok = candidate_camera in CAMERA_ADJACENCY.get(from_camera, [])
        emb_ok = emb_distance < EMBEDDING_DISTANCE_THRESHOLD
        speed_ok = MIN_FEASIBLE_SPEED_KMH <= implied_speed_kmh <= MAX_FEASIBLE_SPEED_KMH
        bearing_ok = bearing_delta <= BEARING_TOLERANCE_DEG

        matched = adjacency_ok and emb_ok and speed_ok and bearing_ok

        reasons = []
        if not adjacency_ok:
            reasons.append("cameras are not adjacent in the network topology")
        if not emb_ok:
            reasons.append(f"embedding distance {emb_distance:.3f} >= threshold {EMBEDDING_DISTANCE_THRESHOLD}")
        if not speed_ok:
            reasons.append(f"implied speed {implied_speed_kmh:.1f} km/h outside feasible window "
                            f"[{MIN_FEASIBLE_SPEED_KMH}, {MAX_FEASIBLE_SPEED_KMH}]")
        if not bearing_ok:
            reasons.append(f"exit bearing deviates {bearing_delta:.0f}° from expected heading "
                            f"(tolerance {BEARING_TOLERANCE_DEG}°)")
        reason = "Identity confirmed — all gates passed." if matched else "Rejected: " + "; ".join(reasons)

        event = HandoverEvent(
            corridor_id=corridor_id,
            from_camera=from_camera,
            to_camera=candidate_camera,
            timestamp=when,
            embedding_distance=round(emb_distance, 4),
            travel_time_s=round(travel_time_s, 2),
            implied_speed_kmh=round(implied_speed_kmh, 1),
            bearing_delta_deg=round(bearing_delta, 1),
            matched=matched,
            reason=reason,
        )
        track.hops.append(event)

        if matched:
            track.current_camera = candidate_camera
            track.embedding = entry_embedding
            track.exit_time = None
            track.exit_bearing = None
            track.confirmed = True
            self.lost_since.pop(corridor_id, None)
        else:
            self.lost_since.setdefault(corridor_id, track.exit_time)

        return event

    def is_lost(self, corridor_id: str, now: Optional[datetime] = None) -> bool:
        """True once handover has failed to confirm beyond HANDOVER_TIMEOUT_S
        (feeds degradation.py's HealthSignal.tracking_ok)."""
        lost_since = self.lost_since.get(corridor_id)
        if not lost_since:
            return False
        now = now or datetime.now()
        return (now - lost_since).total_seconds() > HANDOVER_TIMEOUT_S

    def status(self, corridor_id: str) -> str:
        if corridor_id not in self.tracks:
            return "UNKNOWN"
        if self.is_lost(corridor_id):
            return "LOST — CONSERVATIVE HOLD"
        track = self.tracks[corridor_id]
        return f"TRACKED @ {track.current_camera}" if track.confirmed else "IN TRANSIT"

    def timeline(self, corridor_id: str) -> List[HandoverEvent]:
        track = self.tracks.get(corridor_id)
        return list(track.hops) if track else []


# ─────────────────────────────────────────────
# STANDALONE DEMO
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # emoji-safe on Windows consoles (cp1252)
    except AttributeError:
        pass

    mgr = ReIDHandoverManager()
    corridor_id = "EV-1"
    route = ["West Junction", "Main Junction", "North Junction"]

    mgr.acquire(corridor_id, route[0])
    print(f"Acquired {corridor_id} at {route[0]}. Status: {mgr.status(corridor_id)}\n")

    now = datetime.now()
    for i in range(len(route) - 1):
        from_cam, to_cam = route[i], route[i + 1]
        bearing = junction_bearing_deg(from_cam, to_cam)
        mgr.exit_camera(corridor_id, bearing_deg=bearing, when=now)

        # Plausible transit: ~40 km/h across the inter-camera distance.
        dist_m = junction_distance_m(from_cam, to_cam)
        travel_s = dist_m / (40.0 * 1000 / 3600)
        arrival = now.fromtimestamp(now.timestamp() + travel_s)

        event = mgr.attempt_handover(corridor_id, to_cam, when=arrival)
        print(f"Hop {i+1}: {from_cam} -> {to_cam}")
        print(f"  matched={event.matched}  emb_dist={event.embedding_distance}  "
              f"speed={event.implied_speed_kmh} km/h  bearing_delta={event.bearing_delta_deg}°")
        print(f"  {event.reason}")
        print(f"  Status: {mgr.status(corridor_id)}\n")
        now = arrival

    print("Full timeline:")
    for ev in mgr.timeline(corridor_id):
        print(f"  {ev.from_camera} -> {ev.to_camera}: matched={ev.matched} ({ev.reason})")
