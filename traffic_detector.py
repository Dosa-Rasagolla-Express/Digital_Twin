"""
traffic_detector.py
===================
YOLOv8 vehicle detection module.

Reads a video file (or webcam), detects and tracks vehicles,
estimates speed, detects ambulances, saves latest_frame.jpg,
and writes traffic records to SQLite every few seconds.

Usage
-----
    python traffic_detector.py --source output_ambulance_detection.mp4
    python traffic_detector.py --source 0   # webcam
"""

import cv2
import time
import math
import argparse
import os
from datetime import datetime

import numpy as np
import torch
from ultralytics import YOLO

from database import save_traffic_record, save_alert, init_db
from signal_optimizer import smart_green_time

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
MODEL_PATH       = "yolov8s.pt"
FRAME_SAVE_PATH  = os.path.join(os.path.dirname(__file__), "latest_frame.jpg")
DB_SAVE_INTERVAL = 3          # seconds between DB writes
PIXEL_TO_METER   = 0.05       # rough pixel-to-metre calibration

# COCO class IDs that count as vehicles
VEHICLE_CLASSES = {2, 3, 5, 7}    # car, motorcycle, bus, truck
VEHICLE_NAMES   = {2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck"}

# Extra classes that might be in a custom model
AMBULANCE_KEYWORDS = {"ambulance", "emergency"}

# ─────────────────────────────────────────────
# DEVICE
# ─────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[Detector] Using device: {DEVICE}")


# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────

def load_model(path: str = MODEL_PATH) -> YOLO:
    """Load and return a YOLOv8 model."""
    model = YOLO(path)
    model.to(DEVICE)
    print(f"[Detector] Model loaded from: {path}")
    return model


# ─────────────────────────────────────────────
# SPEED ESTIMATION HELPERS
# ─────────────────────────────────────────────

def estimate_speed(
    obj_id: int,
    cx: int,
    cy: int,
    current_time: float,
    prev_positions: dict,
    prev_times: dict,
) -> float:
    """
    Estimate speed in km/h using centre-point displacement between frames.
    Updates prev_positions and prev_times in-place.
    """
    speed_kmh = 0.0

    if obj_id in prev_positions:
        px, py = prev_positions[obj_id]
        dt     = current_time - prev_times[obj_id]

        if dt > 0:
            dist_px  = math.hypot(cx - px, cy - py)
            dist_m   = dist_px * PIXEL_TO_METER
            speed_kmh = (dist_m / dt) * 3.6

    prev_positions[obj_id] = (cx, cy)
    prev_times[obj_id]     = current_time
    return speed_kmh


# ─────────────────────────────────────────────
# FRAME ANNOTATOR
# ─────────────────────────────────────────────

def draw_traffic_light(frame: np.ndarray, state: str, remaining: int) -> np.ndarray:
    """Draw a traffic-light overlay on the frame."""
    # Background box
    cv2.rectangle(frame, (20, 160), (120, 400), (40, 40, 40), -1)
    cv2.rectangle(frame, (20, 160), (120, 400), (80, 80, 80), 2)

    red_col    = (0, 0, 200) if state == "RED"    else (0, 0, 60)
    yellow_col = (0, 200, 200) if state == "YELLOW" else (0, 60, 60)
    green_col  = (0, 200, 0) if state == "GREEN"  else (0, 60, 0)

    cv2.circle(frame, (70, 210), 28, red_col,    -1)
    cv2.circle(frame, (70, 280), 28, yellow_col, -1)
    cv2.circle(frame, (70, 350), 28, green_col,  -1)

    cv2.putText(
        frame, f"{state}", (22, 405),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2
    )
    cv2.putText(
        frame, f"{remaining}s", (55, 430),
        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 255, 200), 2
    )
    return frame


def annotate_frame(
    frame: np.ndarray,
    vehicle_count: int,
    avg_speed: float,
    signal_state: str,
    congestion: str,
    ambulance: bool,
    remaining: int,
) -> np.ndarray:
    """Overlay dashboard stats on the video frame."""
    frame = draw_traffic_light(frame, signal_state, remaining)

    # HUD background
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 75), (10, 10, 30), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    hud_texts = [
        (f"Vehicles: {vehicle_count}", (10, 25)),
        (f"Speed: {avg_speed:.1f} km/h", (220, 25)),
        (f"Signal: {signal_state}", (10, 55)),
        (f"Congestion: {congestion}", (220, 55)),
    ]
    for text, pos in hud_texts:
        cv2.putText(frame, text, pos,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 230, 255), 2)

    if ambulance:
        cv2.putText(
            frame, "🚑 AMBULANCE DETECTED — PRIORITY ACTIVE",
            (10, frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
        )
        # Red border
        cv2.rectangle(frame, (0, 0),
                      (frame.shape[1]-1, frame.shape[0]-1), (0, 0, 255), 4)

    return frame


# ─────────────────────────────────────────────
# MAIN DETECTION LOOP
# ─────────────────────────────────────────────

def run_detection(source, show_window: bool = True):
    """
    Main detection loop.

    Parameters
    ----------
    source       : video file path (str) or webcam index (int)
    show_window  : whether to display the OpenCV window
    """
    init_db()
    model = load_model()

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[Detector] ❌ Cannot open source: {source}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    print(f"[Detector] ✅ Source opened — FPS={fps:.1f}")

    # Tracking state
    prev_positions: dict  = {}
    prev_times: dict      = {}
    vehicle_speeds: dict  = {}

    # Signal state machine
    signal_state    = "GREEN"
    signal_timer    = 0.0
    green_duration  = 30
    yellow_duration = 5
    red_duration    = 20

    # DB save throttle
    last_db_save = 0.0

    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            # Loop the video for continuous demo
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
            if not ret:
                break

        frame_idx += 1
        now = time.time()

        # ── YOLOv8 Tracking ──────────────────────────────
        results = model.track(frame, persist=True, verbose=False)[0]

        frame_vehicle_count    = 0
        frame_total_speed      = 0.0
        frame_ambulance        = False

        if results.boxes is not None and results.boxes.id is not None:
            boxes   = results.boxes.xyxy.cpu().numpy()
            ids     = results.boxes.id.cpu().numpy().astype(int)
            classes = results.boxes.cls.cpu().numpy().astype(int)
            confs   = results.boxes.conf.cpu().numpy()

            for box, obj_id, cls, conf in zip(boxes, ids, classes, confs):
                class_name = model.names[cls].lower()
                is_vehicle  = cls in VEHICLE_CLASSES
                is_ambulance = any(kw in class_name for kw in AMBULANCE_KEYWORDS)

                if not (is_vehicle or is_ambulance):
                    continue

                x1, y1, x2, y2 = map(int, box)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                spd = estimate_speed(
                    obj_id, cx, cy, now,
                    prev_positions, prev_times
                )
                vehicle_speeds[obj_id] = spd

                frame_vehicle_count += 1
                frame_total_speed   += spd

                if is_ambulance:
                    frame_ambulance = True
                    color = (0, 0, 255)
                    label = f"AMBULANCE | {spd:.0f} km/h"
                else:
                    color = (0, 220, 90)
                    display_name = VEHICLE_NAMES.get(cls, class_name.capitalize())
                    label = f"{display_name} | {spd:.0f} km/h"

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame, label, (x1, max(30, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2
                )

        avg_speed = (
            frame_total_speed / frame_vehicle_count
            if frame_vehicle_count > 0 else 0.0
        )

        # ── Signal Optimization ──────────────────────────
        green_duration, congestion = smart_green_time(
            frame_vehicle_count, avg_speed, frame_ambulance
        )

        signal_timer += 1 / fps

        if signal_state == "GREEN":
            remaining = max(0, int(green_duration - signal_timer))
            if signal_timer >= green_duration:
                signal_state  = "YELLOW"
                signal_timer  = 0.0

        elif signal_state == "YELLOW":
            remaining = max(0, int(yellow_duration - signal_timer))
            if signal_timer >= yellow_duration:
                signal_state  = "RED"
                signal_timer  = 0.0

        else:  # RED
            remaining = max(0, int(red_duration - signal_timer))
            if signal_timer >= red_duration:
                signal_state  = "GREEN"
                signal_timer  = 0.0

        # ── Annotate Frame ───────────────────────────────
        frame = annotate_frame(
            frame, frame_vehicle_count, avg_speed,
            signal_state, congestion, frame_ambulance, remaining
        )

        # ── Save Latest Frame ────────────────────────────
        cv2.imwrite(FRAME_SAVE_PATH, frame)

        # ── Save to Database ─────────────────────────────
        if now - last_db_save >= DB_SAVE_INTERVAL:
            save_traffic_record(
                vehicle_count = frame_vehicle_count,
                avg_speed     = avg_speed,
                signal        = signal_state,
                congestion    = congestion,
                ambulance     = int(frame_ambulance),
            )
            if frame_ambulance:
                save_alert("AMBULANCE", "Ambulance detected — emergency corridor activated")
            if congestion == "HIGH":
                save_alert("CONGESTION", f"High congestion — {frame_vehicle_count} vehicles")
            last_db_save = now
            print(
                f"[{datetime.now():%H:%M:%S}] "
                f"Vehicles={frame_vehicle_count} Speed={avg_speed:.1f} "
                f"Signal={signal_state} Congestion={congestion} "
                f"Ambulance={frame_ambulance}"
            )

        # ── Display Window ───────────────────────────────
        if show_window:
            cv2.imshow("Smart Traffic Digital Twin — Detector", frame)
            if cv2.waitKey(1) & 0xFF == 27:   # ESC to quit
                break

    cap.release()
    if show_window:
        cv2.destroyAllWindows()
    print("[Detector] Detection loop ended.")


# ─────────────────────────────────────────────
# CLI ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smart Traffic Detector")
    parser.add_argument(
        "--source", default="output_ambulance_detection.mp4",
        help="Video file or webcam index (0)"
    )
    parser.add_argument(
        "--no-window", action="store_true",
        help="Disable OpenCV display window (headless mode)"
    )
    args = parser.parse_args()

    src = int(args.source) if args.source.isdigit() else args.source
    run_detection(src, show_window=not args.no_window)
