import time
import cv2
from ultralytics import YOLO
import torch

# Model and device configurations
WEIGHTS = "/home/cagan/yolo-train/runs/detect/train/weights/best.pt"
model = YOLO(WEIGHTS)
device = 0 if torch.cuda.is_available() else 'cpu'
model.to(device)

# Initialize and configure camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Loop for detection
prev = time.time()
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Couldn't open camera.")
        break

    results = model.track(
        frame,
        imgsz=640,
        conf=0.20,
        iou=0.65,
        device=device,
        half=True,                     # False if working with CPU
        tracker="trackers/bytetrack.yaml",  # or "trackers/botsort.yaml"
        persist=True,                  # takip için durumun korunması şart
        verbose=False
    )

    annotated = results[0].plot()

    # Measure fps
    now = time.time()
    fps = 1.0 / (now - prev)
    prev = now
    cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("YOLO Object Tracking", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release and destroy windows
cap.release()
cv2.destroyAllWindows()
