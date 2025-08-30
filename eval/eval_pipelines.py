"""Evaluate two-stage and YOLO pipelines on a labeled image list.

Expect a CSV with 'image,label' per row. The script will run both pipelines and
report accuracy, average inference time, and mean RSS/CPU during runs.
"""
import os
import time
import csv
import psutil
from pathlib import Path
from statistics import mean
from typing import List, Tuple
import numpy as np

import cv2

from two_stage.detector import detect_and_crop
from two_stage.classifier import TinyClassifier, cv2_resize
from yolo.runner import load_model, infer as yolo_infer


def load_labels(csv_path: str, prefix: str = "Data/") -> List[Tuple[str, int]]:
    rows = []
    if not os.path.exists(csv_path):
        print(f"[ERROR] Labels CSV not found: {csv_path}")
        return rows
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        for idx, r in enumerate(reader):
            if not r or len(r) < 2:
                print(f"[WARN] Row {idx} malformed: {r}")
                continue
            try:
                # Prepend the prefix (e.g., "Data/")
                img_path = os.path.join(prefix, r[0]) if prefix else r[0]
                rows.append((img_path, int(r[1])))
            except Exception as e:
                print(f"[WARN] Row {idx} parse error: {r}, {e}")
    return rows


def eval_two_stage(labels_csv: str, classifier: TinyClassifier, input_size=(64, 64)):
    from tqdm import tqdm
    rows = load_labels(labels_csv)
    if not rows:
        print(f"[ERROR] No valid rows in labels CSV: {labels_csv}")
        return {'accuracy': 0, 'avg_latency': 0, 'mean_rss': 0, 'mean_cpu': 0, 'n': 0}
    proc = psutil.Process()
    latencies = []
    rss_samples = []
    cpu_samples = []
    preds = []
    trues = []
    for idx, (path, label) in enumerate(tqdm(rows, desc='Two-stage eval')):
        if not os.path.exists(path):
            print(f"[WARN] Image not found: {path}")
            continue
        img = cv2.imread(path)
        if img is None:
            print(f"[WARN] Failed to read image: {path}")
            continue
        start = time.time()
        crops = detect_and_crop(img)
        crop = crops[0] if crops else None
        if crop is None:
            pred = -1
        else:
            arr = cv2_resize(crop, input_size)
            arr = np.expand_dims(arr.astype('float32') / 255.0, 0)
            if classifier.model is None:
                raise RuntimeError("Classifier model is not built or loaded.")
            try:
                probs = classifier.model.predict(arr, verbose=0)
                pred = int(np.argmax(probs, axis=1)[0])
            except Exception as e:
                print(f"[ERROR] Classifier predict failed for {path}: {e}")
                pred = -1
        latency = time.time() - start
        latencies.append(latency)
        rss_samples.append(proc.memory_info().rss)
        cpu_samples.append(psutil.cpu_percent(interval=None))
        preds.append(pred)
        trues.append(label)
    if not trues:
        print(f"[ERROR] No valid images processed in two-stage eval.")
        return {'accuracy': 0, 'avg_latency': 0, 'mean_rss': 0, 'mean_cpu': 0, 'n': 0}
    acc = sum(1 for p, t in zip(preds, trues) if p == t) / len(trues)
    print(f"[DEBUG] Two-stage: {len(trues)} samples, accuracy={acc:.4f}")
    return {
        'accuracy': acc,
        'avg_latency': mean(latencies),
        'mean_rss': mean(rss_samples),
        'mean_cpu': mean(cpu_samples),
        'n': len(trues)
    }


def eval_yolo(labels_csv: str, yolo_path: str = 'yolov8n.pt', conf: float = 0.25):
    from tqdm import tqdm
    rows = load_labels(labels_csv)
    if not rows:
        print(f"[ERROR] No valid rows in labels CSV: {labels_csv}")
        return {'accuracy': 0, 'avg_latency': 0, 'mean_rss': 0, 'mean_cpu': 0, 'n': 0}
    try:
        model = load_model(yolo_path)
    except Exception as e:
        print(f"[ERROR] Failed to load YOLO model: {e}")
        return {'accuracy': 0, 'avg_latency': 0, 'mean_rss': 0, 'mean_cpu': 0, 'n': 0}
    proc = psutil.Process()
    latencies = []
    rss_samples = []
    cpu_samples = []
    preds = []
    trues = []
    for idx, (path, label) in enumerate(tqdm(rows, desc='YOLO eval')):
        if not os.path.exists(path):
            print(f"[WARN] Image not found: {path}")
            continue
        img = cv2.imread(path)
        if img is None:
            print(f"[WARN] Failed to read image: {path}")
            continue
        start = time.time()
        try:
            detections, latency = yolo_infer(model, img, conf=conf)
        except Exception as e:
            print(f"[ERROR] YOLO inference failed for {path}: {e}")
            detections, latency = [], 0.0
        if not detections:
            pred = -1
        else:
            best = max(detections, key=lambda d: d['conf'])
            pred = int(best['cls'])
        total_latency = time.time() - start
        latencies.append(total_latency)
        rss_samples.append(proc.memory_info().rss)
        cpu_samples.append(psutil.cpu_percent(interval=None))
        preds.append(pred)
        trues.append(label)
    if not trues:
        print(f"[ERROR] No valid images processed in YOLO eval.")
        return {'accuracy': 0, 'avg_latency': 0, 'mean_rss': 0, 'mean_cpu': 0, 'n': 0}
    acc = sum(1 for p, t in zip(preds, trues) if p == t) / len(trues)
    print(f"[DEBUG] YOLO: {len(trues)} samples, accuracy={acc:.4f}")
    return {
        'accuracy': acc,
        'avg_latency': mean(latencies),
        'mean_rss': mean(rss_samples),
        'mean_cpu': mean(cpu_samples),
        'n': len(trues)
    }


def run_comparison(labels_csv: str, classifier: TinyClassifier, yolo_path: str = 'yolov8n.pt'):
    print('Evaluating two-stage pipeline...')
    two_stats = eval_two_stage(labels_csv, classifier, input_size=classifier.input_size)
    print('Evaluating YOLO pipeline...')
    yolo_stats = eval_yolo(labels_csv, yolo_path=yolo_path)
    print('\nResults:')
    print('Two-stage:', two_stats)
    print('YOLO:', yolo_stats)
    return {'two_stage': two_stats, 'yolo': yolo_stats}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_csv', required=True, help='test.csv for two-stage (no bbox)')
    parser.add_argument('--test_converted_csv', required=True, help='test_converted.csv for YOLO (with bbox)')
    parser.add_argument('--yolo', default='yolov8n.pt', help='YOLO model weights')
    parser.add_argument('--classifier', default=None, help='Path to saved tiny classifier (h5 or folder)')
    parser.add_argument('--num_classes', type=int, default=43)
    args = parser.parse_args()

    # load classifier
    clf = TinyClassifier(num_classes=args.num_classes)
    if args.classifier:
        clf.load(args.classifier)
    else:
        clf.build()
        print('Warning: classifier built untrained — results will be meaningless.')
    # Evaluate two-stage pipeline on test.csv (no bbox)
    print('Evaluating two-stage pipeline...')
    two_stats = eval_two_stage(args.test_csv, clf, input_size=clf.input_size)
    # Evaluate YOLO pipeline on test_converted.csv (with bbox)
    print('Evaluating YOLO pipeline...')
    yolo_stats = eval_yolo(args.test_converted_csv, yolo_path=args.yolo)
    print('\nResults:')
    print('Two-stage:', two_stats)
    print('YOLO:', yolo_stats)
