"""Simple YOLOv8 runner using ultralytics package.

Provides helpers to load model, run inference on frames, and evaluate on dataset.
"""
from typing import List, Tuple
import time
import numpy as np


def load_model(path: str = 'yolov8n.pt'):
    import os
    if not os.path.exists(path):
        print(f"[ERROR] YOLO model weights not found: {path}")
        raise FileNotFoundError(path)
    try:
        from ultralytics import YOLO
    except Exception as e:
        print(f"[ERROR] ultralytics import failed: {e}")
        raise RuntimeError('ultralytics package is required: ' + str(e))
    print(f"[INFO] Loading YOLO model from {path}")
    model = YOLO(path)
    return model


from typing import Tuple

def infer(model, frame: np.ndarray, conf: float = 0.25) -> Tuple[List[dict], float]:
    # returns list of detections with xyxy, conf, cls
    import cv2
    if frame is None or not isinstance(frame, np.ndarray):
        print(f"[ERROR] Invalid input frame for YOLO inference.")
        return [], 0.0
    start = time.time()
    try:
        res = model.predict(source=frame, conf=conf, verbose=False)
    except Exception as e:
        print(f"[ERROR] YOLO inference failed: {e}")
        return [], 0.0
    latency = (time.time() - start)
    detections = []
    r = res[0]
    if hasattr(r, 'boxes') and r.boxes is not None:
        boxes = r.boxes.xyxy.cpu().numpy()
        scores = r.boxes.conf.cpu().numpy()
        cls = r.boxes.cls.cpu().numpy().astype(int)
        for b, s, c in zip(boxes, scores, cls):
            detections.append({'xyxy': b, 'conf': float(s), 'cls': int(c)})
    print(f"[DEBUG] YOLO detections: {len(detections)}, latency: {latency:.4f}s")
    return detections, latency
