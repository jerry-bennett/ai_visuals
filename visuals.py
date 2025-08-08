import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import torch
from ultralytics import YOLO
import cv2
import numpy as np
import time

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# Create the named window before starting capture
win_name = "YOLO Detection + Pose (Full Frame, Half Precision)"
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL | cv2.WINDOW_FREERATIO)

# Optional: manually maximize to screen size (macOS-safe)
screen_res = 1920, 1080  # adjust if you know your display resolution
cv2.resizeWindow(win_name, *screen_res)

# Load smaller models and switch to half precision for speed
det_model = YOLO("yolov8n.pt").to(device).half()
pose_model = YOLO("yolov8n-pose.pt").to(device).half()

SKELETON_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (0, 5), (0, 6), (5, 7), (7, 9),
    (6, 8), (8, 10), (5, 6), (5, 11),
    (6, 12), (11, 12), (11, 13), (13, 15),
    (12, 14), (14, 16)
]

PALETTES = [
    [(255, 20, 147), (0, 255, 255), (255, 140, 0)],
    [(0, 255, 0), (255, 255, 0), (0, 191, 255)],
    [(255, 0, 255), (0, 255, 128), (255, 255, 255)]
]

palette_index = 0
current_palette = PALETTES[palette_index]
track_colors = {}

def get_color(track_id):
    if track_id not in track_colors:
        color = current_palette[len(track_colors) % len(current_palette)]
        track_colors[track_id] = color
    return track_colors[track_id]

cap = cv2.VideoCapture(0)
prev_time = time.time()
fps = 0

pose_interval = 3  # Run pose every 3 frames
frame_idx = 0
pose_results = []  # Cache last pose results

while True:
    ret, frame = cap.read()
    if not ret:
        break

    small_frame = cv2.resize(frame, (320, 180))

    with torch.no_grad():
        det_results = det_model(small_frame, verbose=False)
        if frame_idx % pose_interval == 0:
            pose_results = pose_model(small_frame, verbose=False)

    scale_x = frame.shape[1] / small_frame.shape[1]
    scale_y = frame.shape[0] / small_frame.shape[0]

    # Draw detection boxes on full res frame
    for res in det_results:
        boxes = res.boxes.xyxy.cpu().numpy()
        classes = res.boxes.cls.cpu().numpy()
        scores = res.boxes.conf.cpu().numpy()
        for box, cls, score in zip(boxes, classes, scores):
            x1, y1, x2, y2 = box.astype(int)
            x1o, y1o = int(x1 * scale_x), int(y1 * scale_y)
            x2o, y2o = int(x2 * scale_x), int(y2 * scale_y)
            cv2.rectangle(frame, (x1o, y1o), (x2o, y2o), (0, 255, 0), 2)
            label = res.names[int(cls)]
            cv2.putText(frame, f"{label} {score:.2f}", (x1o, y1o - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Draw pose skeletons on full res frame (from cached results)
    for res in pose_results:
        if res.keypoints is None:
            continue
        keypoints = res.keypoints.xy.cpu().numpy()
        track_ids = getattr(res.boxes, 'tracker_id', None)

        for i, person_kpts in enumerate(keypoints):
            track_id = int(track_ids[i]) if track_ids is not None else i
            color = get_color(track_id)

            person_kpts[:, 0] *= scale_x
            person_kpts[:, 1] *= scale_y
            person_kpts = person_kpts.astype(int)

            for a, b in SKELETON_CONNECTIONS:
                cv2.line(frame, tuple(person_kpts[a]), tuple(person_kpts[b]),
                         color, 2, lineType=cv2.LINE_AA)

            for x, y in person_kpts:
                cv2.circle(frame, (x, y), 5, color, -1, lineType=cv2.LINE_AA)

            cv2.putText(frame, f"ID:{track_id}", tuple(person_kpts[0] + [5, -5]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # FPS calc and display
    curr_time = time.time()
    fps = 0.9 * fps + 0.1 * (1 / (curr_time - prev_time))
    prev_time = curr_time
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow(win_name, frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        palette_index = (palette_index + 1) % len(PALETTES)
        current_palette = PALETTES[palette_index]
        track_colors.clear()
        print(f"Switched to palette #{palette_index + 1}")

    frame_idx += 1

cap.release()
cv2.destroyAllWindows()
