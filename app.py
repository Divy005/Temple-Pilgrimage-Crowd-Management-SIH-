import cv2
import numpy as np
from ultralytics import YOLO
from norfair import Detection, Tracker

# -----------------------------
# PARAMETERS
# -----------------------------
VIDEO_SOURCE = 0           # 0 = webcam, or "video.mp4"
YOLO_MODEL_PATH = "yolov8n.pt"
CONF_THRESH = 0.4
DISTANCE_THRESHOLD = 80     # Norfair centroid distance threshold
LINE_POSITION_RATIO = 0.5   # Line at middle of frame (0.0 top, 1.0 bottom)
MAX_LOST = 6                # Frames to keep lost tracks

# -----------------------------
# INITIALIZE DETECTOR AND TRACKER
# -----------------------------
yolo = YOLO(YOLO_MODEL_PATH)
tracker = Tracker(
    distance_function=lambda det, tr: np.linalg.norm(det.points[0] - tr.estimate[0]),
    distance_threshold=DISTANCE_THRESHOLD
)

cap = cv2.VideoCapture(VIDEO_SOURCE)
if not cap.isOpened():
    raise RuntimeError("Cannot open video source")

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
line_y = int(height * LINE_POSITION_RATIO)

counted_ids = set()
total_count = 0

print("Starting people counting. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    display = frame.copy()

    # -----------------------------
    # 1) YOLO detection
    # -----------------------------
    results = yolo.predict(frame, stream=False)
    detections = []

    for result in results:
        if hasattr(result, "boxes") and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()

            for i, cls in enumerate(classes):
                if int(cls) == 0 and scores[i] >= CONF_THRESH:  # person
                    x1, y1, x2, y2 = boxes[i].astype(int)
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                    pt = np.array([[cx, cy]])
                    detections.append(Detection(points=pt, scores=np.array([scores[i]]), data={"bbox": (x1, y1, x2, y2)}))

    # -----------------------------
    # 2) Update tracker
    # -----------------------------
    tracked_objects = tracker.update(detections)

    # -----------------------------
    # 3) Count people crossing line
    # -----------------------------
    for obj in tracked_objects:
        cx, cy = obj.estimate[0]
        tid = obj.id

        # Count only if not already counted and crossing line downward
        if tid not in counted_ids and obj.last_detection is not None:
            last_cy = obj.last_detection.points[0][1]
            if last_cy < line_y <= cy:  # moving downward across line
                total_count += 1
                counted_ids.add(tid)

    # -----------------------------
    # 4) Draw results
    # -----------------------------
    cv2.line(display, (0, line_y), (width, line_y), (0, 0, 255), 2)
    cv2.putText(display, f"Total Count: {total_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    for obj in tracked_objects:
        x1, y1, x2, y2 = obj.last_detection.data["bbox"]
        cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(display, f"ID {obj.id}", (x1, max(0, y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
        cx, cy = obj.estimate[0]
        cv2.circle(display, (int(cx), int(cy)), 3, (255, 0, 0), -1)

    cv2.imshow("People Counting - Top Down", display)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
