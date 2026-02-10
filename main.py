import cv2
import time
import math
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.models import resnet18
from ultralytics import YOLO

# --------------------------------
# DEVICE
# --------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --------------------------------
# LOAD YOLO MODEL
# --------------------------------
yolo_model = YOLO("yolov8s.pt")

# COCO Vehicle Classes
VEHICLE_CLASSES = [2, 3, 5, 7]

# --------------------------------
# LOAD RESNET AMBULANCE MODEL
# --------------------------------
resnet = resnet18()
resnet.fc = torch.nn.Linear(resnet.fc.in_features, 2)

resnet.load_state_dict(torch.load("ambulance_resnet.pth", map_location=DEVICE))
resnet.to(DEVICE)
resnet.eval()

# --------------------------------
# IMAGE TRANSFORM FOR RESNET
# --------------------------------
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# --------------------------------
# VIDEO SOURCE
# --------------------------------
cap = cv2.VideoCapture("videos/Ambulance.mp4")

PIXEL_TO_METER = 0.05

previous_positions = {}
previous_times = {}
vehicle_speeds = {}

# --------------------------------
# MAIN LOOP
# --------------------------------
while cap.isOpened():

    ret, frame = cap.read()
    if not ret:
        break

    results = yolo_model.track(frame, persist=True, verbose=False)[0]

    if results.boxes is not None and results.boxes.id is not None:

        boxes = results.boxes.xyxy.cpu().numpy()
        ids = results.boxes.id.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy()

        for box, obj_id, cls in zip(boxes, ids, classes):

            if int(cls) not in VEHICLE_CLASSES:
                continue

            x1, y1, x2, y2 = map(int, box)

            # -------- SPEED CALCULATION --------
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            obj_id = int(obj_id)
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
                    speed = distance_meters / time_diff
                    speed_kmh = speed * 3.6

            vehicle_speeds[obj_id] = int(speed_kmh)

            previous_positions[obj_id] = (center_x, center_y)
            previous_times[obj_id] = current_time

            # -------- AMBULANCE CLASSIFICATION --------
            vehicle_crop = frame[y1:y2, x1:x2]

            label = "Vehicle"
            color = (0, 255, 0)

            if vehicle_crop.size != 0:
                try:
                    img = transform(vehicle_crop).unsqueeze(0).to(DEVICE)

                    with torch.no_grad():
                        output = resnet(img)

                        prob = F.softmax(output, dim=1)
                        confidence, pred = torch.max(prob, dim=1)

                        confidence = confidence.item()
                        pred = pred.item()

                    # Only label ambulance if confidence high
                    if pred == 0 and confidence > 0.85:
                        label = f"AMBULANCE 🚑 ({confidence:.2f})"
                        color = (0, 0, 255)

                except:
                    pass

            # -------- DRAW BOX --------
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            text = f"{label} | {vehicle_speeds[obj_id]} km/h"

            cv2.putText(frame,
                        text,
                        (x1, max(30, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        color,
                        2)

    cv2.imshow("Emergency Traffic AI", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
