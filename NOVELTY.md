# Novelty of the Invention — IECS

> Source: `Patent_draft_REVISED_v2.docx` (Inventive Disclosure, Section 6 & Section 9), condensed here for the project repo.

**Invention:** System and Method for Vision-Based Emergency Vehicle Green Corridor Generation Using a Real-Time Digital Twin Decision Engine and GPS-Free Cross-Camera Vehicle Identity Handover — *Intelligent Emergency Corridor System (IECS)*.

---

## Gap Filled in Prior Art

Every individual ingredient (camera-based ambulance detection, GPS-based corridors, ML-triggered EVP, offline digital twins, adaptive control like SCOOT/SCATS/SURTRAC) exists separately in prior art or products — but no disclosed system unifies them into one **passive, closed-loop, city-scale operational pipeline**. Specific gaps:

- **Twin-as-decision-engine gap** — all known digital-twin EVP usages are offline evaluators or visualizations, never a live executor.
- **GPS-free continuity gap** — no prior emergency-corridor system tracks ambulance identity across cameras without vehicle-mounted hardware.
- **Uncertainty-aware triggering gap** — prior systems use point-estimate ETAs or learned regressors with no reliability guarantee.
- **Convergent-emergency gap** — no prior system arbitrates overlapping corridors from multiple simultaneous emergencies.
- **Safety-constraint gap** — naïve preemption truncates pedestrian phases; no prior system hard-constrains this inside the optimizer.

The inventive step is the **non-obvious interaction** of these elements: the twin cannot schedule corridors without handover-localized tracking; handover without probabilistic ETAs yields unstable triggering; triggering without twin simulation causes unbounded cross-street harm. The combination is more than the sum of its parts.

---

## Novelty Points (N1–N7)

| # | Novelty | What prior art does instead |
|---|---|---|
| **N1** | **Real-time digital twin as the corridor decision engine.** The twin generates, simulates, scores, and selects whole-corridor signal schedules against live synchronized state during an active emergency, then hands the winning schedule to controllers. | Prior twins (MLEVP, ORNL) evaluate strategies offline or merely visualize state. |
| **N2** | **GPS-free cross-camera identity handover (ReID chain-of-custody).** Unique ambulance ID maintained across successive intersection cameras via embedding matching gated by exit-bearing/entry-gate/travel-time feasibility — zero ambulance-side hardware. | No prior emergency-corridor disclosure implements this. |
| **N3** | **Uncertainty-aware corridor triggering.** Preemption fires on upper-confidence-bound arrival time minus estimated queue-clearance duration, with tunable margins and fail-safe holds. | Denso-type systems use point ETAs; MLEVP uses learned regressors — neither gives a reliability guarantee. |
| **N4** | **Constraint-hardened schedule optimization.** Pedestrian minimum-walk phases, forced yellow, and all-red intervals are hard constraints inside the twin optimizer, producing regulator-compliant schedules by construction. | All cited references use naïve green overrides. |
| **N5** | **Multi-emergency convergence arbitration.** Deterministic merging/sequencing of overlapping corridors by severity class, ETA, and route overlap. | Absent from all located prior art. |
| **N6** | **Graceful degradation hierarchy.** Edge-autonomous fallback (cached corridor + local single-intersection preemption) preserves life-critical function when the cloud/backhaul link fails. | No disclosed counterpart. |
| **N7** | **Passive end-to-end integration.** Detection + handover + prediction + twin scheduling + execution run entirely on existing cameras/controllers — no RFID, GPS, optical/strobe, or DSRC equipment anywhere. | Cost/scalability differentiator vs. every hardware-dependent system cited (RFID, GPS/EVP services, optical/strobe). |

---

## Inventive-Step Argument

Nearest prior-art combinations:
- **IN202641051182 / XAI-adaptive filings** — vision + adaptive signals, no twin, no prediction, no handover.
- **IN202631063464** — GPS/IoT corridor, no vision, no twin, no uncertainty modeling.
- **MLEVP / IWCTS 2024 (Roy et al.)** — ML trigger timing + *offline* twin, GPS/CV-dependent, per-intersection (not corridor-scale) triggers.

Achieving the claimed result — corridor-scale ambulance delay reduction with *bounded* cross-street impact — requires the specific interaction of **N1 + N2 + N3 + N4**. A skilled person starting from any single reference would not arrive at the combination without inventive effort, because each element presupposes design choices incompatible with the others (e.g., MLEVP's queue estimates assume GPS/connected-vehicle penetration that a passive-camera-only system like IECS cannot assume, so its trigger formulation cannot simply be ported over).

---

## Mapping to This Codebase

| Novelty point | Related module(s) in this repo |
|---|---|
| N1 (twin decision engine) | [`twin_engine.py`](twin_engine.py) — implemented: `generate_candidates()` (Candidate Schedule Generator, N schedules varying per-junction offset/green-extension), `simulate_candidate()` (Twin Simulator — per-candidate ambulance/conflicting delay + stop count), `select_schedule()` (Schedule Selector — argmin of J subject to hard constraints). Built on live state from [`network_model.py`](network_model.py), [`simulation_engine.py`](simulation_engine.py), [`eta_predictor.py`](eta_predictor.py). |
| N2 (cross-camera ReID handover) | [`reid_handover.py`](reid_handover.py) — implemented: `ReIDHandoverManager` mints a corridor ID at first detection and confirms it at each adjacent camera only when embedding distance, exit-bearing/entry-gate compatibility, and travel-time feasibility all pass (patent Claim 2); `is_lost()` exposes handover-timeout status to [`degradation.py`](degradation.py). |
| N3 (uncertainty-aware ETA triggering) | [`eta_predictor.py`](eta_predictor.py) — implemented: `predict_routes()` (route-probability distribution), `eta_bands_for_route()` (mean/std/UCB arrival-time bands with compounding uncertainty per hop), `trigger_decision()` (fires when `UCB(ETA) - t_clear(queue) <= margin`, patent Step 6). |
| N4 (constraint-hardened scheduling) | [`twin_engine.py`](twin_engine.py) — `_feasible()` hard-rejects (not merely penalizes) any candidate violating minimum pedestrian walk, yellow, or all-red intervals before scoring, so the selected schedule is regulator-compliant by construction. Legacy rule-based logic remains in [`signal_optimizer.py`](signal_optimizer.py) for the non-emergency dashboard tabs. |
| N5 (multi-EV arbitration) | [`arbitration.py`](arbitration.py) — implemented: merge-compatible corridors via union-find clustering, priority-score ranking (`P = f(severity, ETA, overlap)`), partial segment-wise degradation for losers. Wired into the "🚑 Multi-EV & Resilience" dashboard tab. |
| N6 (degradation hierarchy) | [`degradation.py`](degradation.py) — implemented: `DegradationManager` state machine (`NORMAL` → `REGIONAL_EDGE` → `CONSERVATIVE_HOLD` → `RESTORING`) driven by `HealthSignal` (cloud-link health and — now also fed from `reid_handover.py`'s handover-loss signal — tracking health). Wired into the same dashboard tab, with sidebar toggles to simulate link/tracking loss. |
| N7 (passive integration) | [`traffic_detector.py`](traffic_detector.py), [`network_model.py`](network_model.py), [`database.py`](database.py), [`reid_handover.py`](reid_handover.py) |

*(This table is a rough correspondence for engineering reference, not a legal claim mapping. N1-N3 use simulated embeddings/state rather than a real CV/camera backend — see each module's docstring — but the matching/gating/optimization logic that constitutes the inventive step is real and runnable end-to-end.)*

## N1 / N2 / N3 module reference

- **`reid_handover.py`** — `ReIDHandoverManager.acquire()` mints a corridor ID; `exit_camera()` records the exit bearing; `attempt_handover()` gates the next camera's candidate detection on embedding distance, bearing compatibility, and travel-time feasibility, returning a `HandoverEvent`. Run `python reid_handover.py` for a 3-camera standalone demo.
- **`eta_predictor.py`** — `predict_routes()` scores downstream path hypotheses into a probability distribution; `eta_bands_for_route()` returns mean/std/UCB `ETABand`s per downstream junction; `trigger_decision()` / `trigger_decisions_for_route()` apply the UCB-minus-queue-clear trigger rule. Run `python eta_predictor.py` for a standalone demo.
- **`twin_engine.py`** — `generate_candidates()` → `simulate_candidate()` → `select_schedule()` is the full generate/simulate/select pipeline; hard constraints are enforced in `_feasible()`, never as a soft penalty. Run `python twin_engine.py` for a standalone demo.
- All three are demonstrated live in `dashboard.py`'s **🧠 Twin Decision Engine** tab — enable *Emergency Mode* in the sidebar to see route prediction, ETA/trigger tables, the ranked candidate-schedule table with the twin's selected winner highlighted, and the ReID handover trace, all update together.

## N5 / N6 module reference

- **`arbitration.py`** — `EmergencyVehicle` (route, severity, ETA) → `arbitrate(vehicles)` → list of `ArbitrationDecision` (`FULL` / `MERGED` / `PARTIAL`, priority score, activated segment, reason). Run `python arbitration.py` for a 3-vehicle standalone demo.
- **`degradation.py`** — `DegradationManager.evaluate(HealthSignal(...))` advances the state machine each replan tick; `status_line()` / `recent_history()` expose current mode and the transition log. Run `python degradation.py` for a standalone demo.
- Both are demonstrated live in `dashboard.py`'s **🚑 Multi-EV & Resilience** tab — toggle *Emergency Mode*, the *Convergent Ambulances* slider, *Simulate Cloud Link Failure*, and *Simulate Tracking Loss* in the sidebar to exercise them.
