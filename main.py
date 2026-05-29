import cv2
import time
import math
import torch
from ultralytics import YOLO
from smart_traffic import smart_green_time
from digital_twin import TrafficTwin
from database import save_twin
# --------------------------------
# DEVICE
# --------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --------------------------------
# LOAD YOLO MODEL
# --------------------------------
yolo_model = YOLO("yolov8s.pt")
twin = TrafficTwin()

# COCO vehicle classes
VEHICLE_CLASSES = [2, 3, 5, 7]  # car, motorcycle, bus, truck

# --------------------------------
# VIDEO SOURCE
# --------------------------------
cap = cv2.VideoCapture("output_ambulance_detection.mp4")

if not cap.isOpened():
    print("❌ Error: Video file not opened")
    exit()
else:
    print("✅ Video opened successfully")


PIXEL_TO_METER = 0.05

previous_positions = {}
previous_times = {}
vehicle_speeds = {}

# --------------------------------
# TRAFFIC SIGNAL VARIABLES
# --------------------------------
signal_state = "GREEN"
signal_timer = 0
yellow_duration = 5
red_duration = 20
green_duration = 20

# --------------------------------
# MAIN LOOP
# --------------------------------
while cap.isOpened():

    ret, frame = cap.read()

    if not ret:
       print("❌ Frame not read. End of video or decoding problem.")
       break
    else:
       print("Frame read successfully")


    frame_vehicle_count = 0
    frame_total_speed = 0
    frame_ambulance_detected = False

    # Track vehicles
    results = yolo_model.track(frame, persist=True, verbose=False)[0]

    if results.boxes is not None and results.boxes.id is not None:

        boxes = results.boxes.xyxy.cpu().numpy()
        ids = results.boxes.id.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy()

        for box, obj_id, cls in zip(boxes, ids, classes):

            if int(cls) not in VEHICLE_CLASSES:
                continue

            x1, y1, x2, y2 = map(int, box)
            obj_id = int(obj_id)

            # ---------------- SPEED CALCULATION ----------------
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            current_time = time.time()
            speed_kmh = 0

            if obj_id in previous_positions:
                prev_x, prev_y = previous_positions[obj_id]
                prev_time = previous_times[obj_id]

                distance_pixels = math.sqrt(
                    (center_x - prev_x) ** 2 +
                    (center_y - prev_y) ** 2
                )

                distance_meters = distance_pixels * PIXEL_TO_METER
                time_diff = current_time - prev_time

                if time_diff > 0:
                    speed_kmh = (distance_meters / time_diff) * 3.6

            previous_positions[obj_id] = (center_x, center_y)
            previous_times[obj_id] = current_time
            vehicle_speeds[obj_id] = int(speed_kmh)

            frame_vehicle_count += 1
            frame_total_speed += speed_kmh

            # ---------------- LABELING ----------------
            label = yolo_model.names[int(cls)]
            color = (0, 255, 0)

            # Emergency override if ambulance class detected
            if "ambulance" in label.lower():
                label = "AMBULANCE 🚑"
                color = (0, 0, 255)
                frame_ambulance_detected = True

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            text = f"{label} | {vehicle_speeds[obj_id]} km/h"
            cv2.putText(frame,
                        text,
                        (x1, max(30, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        color,
                        2)

    # --------------------------------
    # SMART TRAFFIC CALCULATION
    # --------------------------------
    if frame_vehicle_count > 0:
        avg_speed = frame_total_speed / frame_vehicle_count
    else:
        avg_speed = 0

    calculated_green_time, congestion = smart_green_time(
        frame_vehicle_count,
        avg_speed,
        frame_ambulance_detected
    )
    twin.update(
    frame_vehicle_count,
    avg_speed,
    signal_state,
    congestion,
    frame_ambulance_detected
)
    save_twin(twin)

    twin.display()

    # --------------------------------
    # SIGNAL STATE MACHINE
    # --------------------------------
    signal_timer += 1 / 30  # assuming ~30 FPS

    if signal_state == "GREEN":
        green_duration = calculated_green_time
        if signal_timer >= green_duration:
            signal_state = "YELLOW"
            signal_timer = 0

    elif signal_state == "YELLOW":
        if signal_timer >= yellow_duration:
            signal_state = "RED"
            signal_timer = 0

    elif signal_state == "RED":
        if signal_timer >= red_duration:
            signal_state = "GREEN"
            signal_timer = 0

    # --------------------------------
    # DRAW TRAFFIC LIGHT
    # --------------------------------
    cv2.rectangle(frame, (20, 200), (120, 400), (50, 50, 50), -1)

    red_color = (0, 0, 100)
    yellow_color = (0, 100, 100)
    green_color = (0, 100, 0)

    if signal_state == "RED":
        red_color = (0, 0, 255)
    elif signal_state == "YELLOW":
        yellow_color = (0, 255, 255)
    elif signal_state == "GREEN":
        green_color = (0, 255, 0)

    cv2.circle(frame, (70, 240), 25, red_color, -1)
    cv2.circle(frame, (70, 300), 25, yellow_color, -1)
    cv2.circle(frame, (70, 360), 25, green_color, -1)

    if signal_state == "GREEN":
        remaining = int(green_duration - signal_timer)
    elif signal_state == "YELLOW":
        remaining = int(yellow_duration - signal_timer)
    else:
        remaining = int(red_duration - signal_timer)

    cv2.putText(frame,
                f"{signal_state}: {remaining}s",
                (20, 430),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2)

    cv2.imshow("Smart Emergency Traffic AI", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

