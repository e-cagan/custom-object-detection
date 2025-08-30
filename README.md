# YOLOv8 Real-Time Object Detection (Store Products)

This project is prepared for real-time detection of store products using **YOLOv8**. The model is trained on the **Roboflow Items Dataset by study-group** and integrated with OpenCV for live camera detect.

---

## Project Structure

```
/home/cagan/yolo-train
├── train.py          # Training script (YOLOv8)
├── detect.py      # Real-time detect (webcam/video)
├── requirements.txt
├── .gitignore
└── README.md
```

Files:
- train.py → Training workflow with YOLOv8
- detect.py → Real-time detection from webcam or video
- requirements.txt → Python dependencies

---

## Installation

Install required packages:

```bash
pip install ultralytics opencv-python
```

For GPU support:
- Ensure your system has the appropriate NVIDIA driver and CUDA installed.
- Install the PyTorch GPU build (example for cu118):

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

## Training

To start training:

```bash
python3 train.py --data data.yaml --model yolov8n.pt --imgsz 512 --epochs 55
```

- data.yaml: dataset configuration
- model: initial weights (e.g. yolov8n.pt)
- imgsz: input size (pixels)
- epochs: number of training epochs

Training results are saved by default under:
runs/detect/train/weights/best.pt

---

## Real-Time detect

To run detection with a webcam or video:

```bash
python3 detect.py --source 0 --weights runs/detect/train/weights/best.pt --imgsz 512
```

- --source 0 → default webcam; provide a file path for a video file
- Output shows class labels and bounding boxes
- Press q to quit

---

## Performance (Example)

- mAP@50: ~93%
- mAP@50-95: ~75%
- FPS: ~20–25 (RTX 3050 Ti, 512px input)

Actual performance varies with hardware and settings.

---

## Notes

- .gitignore prevents large files (dataset, runs, venv, etc.) from being added to the repo.
- Model weights (best.pt) are typically not included in the repository; share separately.
- Dataset source: [Roboflow Items Dataset](https://universe.roboflow.com/study-group/items-balno?utm_source=chatgpt.com)

---

Usage examples, hyperparameters, or inference images can be added if needed.
├── detect.py      # Gerçek zamanlı çıkarım (webcam/video)
├── requirements.txt
├── .gitignore
└── README.md
```