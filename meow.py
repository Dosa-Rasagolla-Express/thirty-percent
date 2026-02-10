import cv2
import time
import math
import torch
import json
import csv
from ultralytics import YOLO

# --------------------------------
# 1. MODEL SETUP
# --------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

base_model = YOLO("yolov8s.pt") 
ambulance_model = YOLO("ambulance_best.pt")

ALL_VEHICLE_CLASSES = [2, 3, 5, 7]   # car, bike, bus, truck
BIG_VEHICLE_CLASSES = [7]            # run ambulance verification only on trucks/big vans

# --------------------------------
# 2. VIDEO + TRACKING SETUP
# --------------------------------
cap = cv2.VideoCapture("videos/traffic.mp4")

previous_positions = {}
previous_times = {}

PIXEL_TO_METER = 0.05  # Approximate scaling

# --------------------------------
# 3. LOG STORAGE
# --------------------------------
frame_logs = []
vehicle_summary = {}

# --------------------------------
# 4. PROCESS VIDEO
# --------------------------------
while cap.isOpened():

    ret, frame = cap.read()
    if not ret:
        break

    frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    timestamp = time.time()

    # --- STAGE 1: Detect & Track Vehicles ---
    results = base_model.track(
        frame,
        persist=True,
        verbose=False,
        classes=ALL_VEHICLE_CLASSES
    )[0]

    if results.boxes is not None and results.boxes.id is not None:

        boxes = results.boxes.xyxy.cpu().numpy()
        ids = results.boxes.id.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy()

        for box, obj_id, cls_idx in zip(boxes, ids, classes):

            x1, y1, x2, y2 = map(int, box)
            obj_id = int(obj_id)
            cls_idx = int(cls_idx)

            # --------------------------------
            # SPEED CALCULATION
            # --------------------------------
            center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            speed_kmh = 0

            if obj_id in previous_positions:

                dist = math.sqrt(
                    (center[0] - previous_positions[obj_id][0]) ** 2 +
                    (center[1] - previous_positions[obj_id][1]) ** 2
                )

                time_diff = timestamp - previous_times[obj_id]

                if time_diff > 0:
                    speed_kmh = (dist * PIXEL_TO_METER / time_diff) * 3.6

            previous_positions[obj_id] = center
            previous_times[obj_id] = timestamp

            # --------------------------------
            # DEFAULT LABEL
            # --------------------------------
            vehicle_type = base_model.names[cls_idx]
            is_ambulance = False
            color = (0, 255, 0)

            # --------------------------------
            # STAGE 2: AMBULANCE VERIFICATION
            # --------------------------------
            if cls_idx in BIG_VEHICLE_CLASSES:

                crop = frame[y1:y2, x1:x2]

                if crop.size != 0:

                    amb_result = ambulance_model.predict(
                        crop,
                        conf=0.80,
                        verbose=False
                    )[0]

                    if len(amb_result.boxes) > 0:

                        is_ambulance = True
                        vehicle_type = "AMBULANCE"
                        color = (0, 0, 255)

            # --------------------------------
            # DRAW VISUALS
            # --------------------------------
            label_text = f"{vehicle_type} | ID:{obj_id} | {int(speed_kmh)} km/h"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label_text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # --------------------------------
            # FRAME LOG ENTRY (JSON)
            # --------------------------------
            frame_logs.append({
                "frame": frame_number,
                "vehicle_id": obj_id,
                "vehicle_type": vehicle_type,
                "speed_kmh": speed_kmh,
                "is_ambulance": is_ambulance,
                "bbox": [x1, y1, x2, y2],
                "timestamp": timestamp
            })

            # --------------------------------
            # VEHICLE SUMMARY
            # --------------------------------
            if obj_id not in vehicle_summary:

                vehicle_summary[obj_id] = {
                    "type": vehicle_type,
                    "speeds": [],
                    "ambulance": False,
                    "first_seen": timestamp,
                    "last_seen": timestamp
                }

            vehicle_summary[obj_id]["speeds"].append(speed_kmh)
            vehicle_summary[obj_id]["last_seen"] = timestamp

            if is_ambulance:
                vehicle_summary[obj_id]["ambulance"] = True

    # --------------------------------
    # SHOW OUTPUT
    # --------------------------------
    cv2.imshow("Traffix-AI", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

# --------------------------------
# 5. SAVE JSON LOG
# --------------------------------
with open("vehicle_frame_logs.json", "w") as f:
    json.dump(frame_logs, f, indent=4)

# --------------------------------
# 6. SAVE VEHICLE SUMMARY CSV
# --------------------------------
with open("vehicle_summary.csv", "w", newline="") as f:

    writer = csv.writer(f)

    writer.writerow([
        "Vehicle_ID",
        "Type",
        "Average_Speed",
        "Max_Speed",
        "Ambulance",
        "Tracking_Duration"
    ])

    for vid, data in vehicle_summary.items():

        avg_speed = sum(data["speeds"]) / len(data["speeds"])
        max_speed = max(data["speeds"])
        duration = data["last_seen"] - data["first_seen"]

        writer.writerow([
            vid,
            data["type"],
            round(avg_speed, 2),
            round(max_speed, 2),
            data["ambulance"],
            round(duration, 2)
        ])

print("✅ Logs saved successfully")
