from ultralytics import YOLO

# Constants
DATA_YAML = "/home/cagan/yolo-train/Items.v2i.yolov8/data.yaml"
BASE_WEIGHTS = "yolov8n.pt"          # istersen 'yolov8s.pt'

# Load the model
model = YOLO(BASE_WEIGHTS)

# Train the model
results = model.train(
    data=DATA_YAML,
    model=BASE_WEIGHTS,
    epochs=100,
    imgsz=512,
    batch=16,
    device=0,
    fraction=1,
    cos_lr=True,
    deterministic=True, # Disables randomness sources for same randomness
    verbose=True,
)

# --- Doğrulama (val split) ---
# (metrics.box.map -> mAP@0.50, metrics.box.map50_95 -> mAP@0.50:0.95)
metrics = model.val(data=DATA_YAML, split="val", imgsz=512, verbose=True)
print(f"[VAL] mAP50={metrics.box.map:.3f} | mAP50-95={metrics.box.map50_95:.3f} | P={metrics.box.mp:.3f} | R={metrics.box.mr:.3f}")

