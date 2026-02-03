import cv2
import time
import math
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

VEHICLE_CLASSES = [2, 3, 5, 7]

cap = cv2.VideoCapture("videos/traffic.mp4")

# Approximate pixel to meter conversion
PIXEL_TO_METER = 0.05   # Adjust based on camera

previous_positions = {}
previous_times = {}
vehicle_speeds = {}

while cap.isOpened():

    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(frame, persist=True, verbose=False)[0]

    if results.boxes.id is not None:

        boxes = results.boxes.xyxy.cpu().numpy()
        ids = results.boxes.id.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy()

        for box, obj_id, cls in zip(boxes, ids, classes):

            if int(cls) not in VEHICLE_CLASSES:
                continue

            x1, y1, x2, y2 = map(int, box)
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            obj_id = int(obj_id)

            current_time = time.time()

            # Default speed
            speed_kmh = 0

            # If we saw vehicle before
            if obj_id in previous_positions:

                prev_x, prev_y = previous_positions[obj_id]
                prev_time = previous_times[obj_id]

                # Pixel displacement
                distance_pixels = math.sqrt(
                    (center_x - prev_x) ** 2 +
                    (center_y - prev_y) ** 2
                )

                # Convert to meters
                distance_meters = distance_pixels * PIXEL_TO_METER

                time_diff = current_time - prev_time

                if time_diff > 0:
                    speed = distance_meters / time_diff
                    speed_kmh = speed * 3.6

            vehicle_speeds[obj_id] = int(speed_kmh)

            # Update previous values
            previous_positions[obj_id] = (center_x, center_y)
            previous_times[obj_id] = current_time

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

            # Draw speed
            text = f"{vehicle_speeds[obj_id]} km/h"
            cv2.putText(frame,
                        text,
                        (x1, max(30, y1-10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0,0,255),
                        2)

    cv2.imshow("Vehicle Speed Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
