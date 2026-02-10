import cv2
import time
import math
import torch
from ultralytics import YOLO

# --------------------------------
# 1. SETUP MODELS
# --------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Base model to detect ALL standard vehicles
base_model = YOLO("yolov8s.pt") 
# Your custom ambulance detection model
ambulance_model = YOLO("ambulance_best.pt")

# Standard COCO classes for vehicles: 2: car, 3: motorcycle, 5: bus, 7: truck
ALL_VEHICLE_CLASSES = [2, 3, 5, 7]
# Only check these for the secondary ambulance classification
BIG_VEHICLE_CLASSES = [7]

# --------------------------------
# 2. VIDEO SOURCE & TRACKING SETUP
# --------------------------------
cap = cv2.VideoCapture("videos/Ambulance.mp4")

previous_positions = {}
previous_times = {}
PIXEL_TO_METER = 0.05 

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    # --- STAGE 1: Track ALL vehicle types ---
    results = base_model.track(frame, persist=True, verbose=False, classes=ALL_VEHICLE_CLASSES)[0]

    if results.boxes is not None and results.boxes.id is not None:
        boxes = results.boxes.xyxy.cpu().numpy()
        ids = results.boxes.id.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy()

        for box, obj_id, cls_idx in zip(boxes, ids, classes):
            x1, y1, x2, y2 = map(int, box)
            obj_id = int(obj_id)
            cls_idx = int(cls_idx)

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

            # Default label and color for general traffic
            label = base_model.names[cls_idx]
            color = (0, 255, 0) # Green

            # --- STAGE 2: High-Confidence Ambulance Check for Big Vehicles ---
            if cls_idx in BIG_VEHICLE_CLASSES:
                vehicle_crop = frame[y1:y2, x1:x2]
                if vehicle_crop.size != 0:
                    # Run custom model with high confidence requirement
                    amb_results = ambulance_model.predict(vehicle_crop, conf=0.80, verbose=False)[0]
                    
                    if len(amb_results.boxes) > 0:
                        # Grab confidence of the best ambulance detection
                        conf_score = amb_results.boxes.conf[0].item()
                        label = f"AMBULANCE 🚑 ({conf_score:.2f})"
                        color = (0, 0, 255) # Red

            # -------- DRAWING --------
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} | {int(speed_kmh)} km/h", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("Traffix-AI: Emergency Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27: break 

cap.release()
cv2.destroyAllWindows()