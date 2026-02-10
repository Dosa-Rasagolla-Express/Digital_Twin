import cv2
import time
import math
import torch
from ultralytics import YOLO

# --------------------------------
# 1. SETUP MODELS
# --------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Primary model (COCO) to find general vehicles
base_model = YOLO("yolov8s.pt") 

# Your custom ambulance model
ambulance_model = YOLO("ambulance_best.pt")

# COCO indices for big vehicles where an ambulance might be "hidden" or misclassified
# 5: bus, 7: truck (Standard COCO indices)
BIG_VEHICLE_CLASSES = [5, 7]

# --------------------------------
# 2. VIDEO SOURCE & TRACKING SETUP
# --------------------------------
video_path = "videos/traffic.mp4"
cap = cv2.VideoCapture(video_path)

# Tracking variables for speed
previous_positions = {}
previous_times = {}
PIXEL_TO_METER = 0.05 

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    # --- STAGE 1: Detect and Track with Base Model ---
    # We only care about Bus (5) and Truck (7) for Stage 2
    results = base_model.track(frame, persist=True, verbose=False, classes=BIG_VEHICLE_CLASSES)[0]

    if results.boxes is not None and results.boxes.id is not None:
        boxes = results.boxes.xyxy.cpu().numpy()
        ids = results.boxes.id.cpu().numpy()

        for box, obj_id in zip(boxes, ids):
            x1, y1, x2, y2 = map(int, box)
            obj_id = int(obj_id)

            # -------- SPEED CALCULATION --------
            current_time = time.time()
            center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            speed_kmh = 0

            if obj_id in previous_positions:
                dist = math.sqrt((center[0]-previous_positions[obj_id][0])**2 + (center[1]-previous_positions[obj_id][1])**2)
                time_diff = current_time - previous_times[obj_id]
                if time_diff > 0:
                    speed_kmh = (dist * PIXEL_TO_METER / time_diff) * 3.6

            previous_positions[obj_id], previous_times[obj_id] = center, current_time

            # --- STAGE 2: Custom Ambulance Check on the Crop ---
            vehicle_crop = frame[y1:y2, x1:x2]
            label = "Big Vehicle"
            color = (0, 255, 0) # Green for normal vehicles

            if vehicle_crop.size != 0:
                # Run your custom model only on the detected bus/truck area
                amb_results = ambulance_model.predict(vehicle_crop, conf=0.7, verbose=False)[0]
                
                # If your custom model finds an 'ambulance' class
                if len(amb_results.boxes) > 0:
                    label = "AMBULANCE 🚑"
                    color = (0, 0, 255) # Red for emergency

            # -------- DRAWING --------
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} | {int(speed_kmh)} km/h", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("Traffix-AI: Emergency Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27: break # ESC to quit

cap.release()
cv2.destroyAllWindows()