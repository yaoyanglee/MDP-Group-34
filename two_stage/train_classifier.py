"""Train TinyClassifier on pre-cropped images and labels.

Usage:
    python two_stage/train_classifier.py \
        --csv processed/classifier/classifier_labels.csv \
        --out processed/classifier/tiny_classifier.h5 \
        --input_size 64 64 \
        --epochs 20 \
        --val_split 0.1

- Validates all inputs and prints summary stats.
- Uses stratified split for train/val.
- Saves model and training log.
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from classifier import TinyClassifier, cv2_resize
import cv2


def validate_inputs(csv_path, input_size):
    if not os.path.exists(csv_path):
        print(f"[ERROR] CSV not found: {csv_path}")
        sys.exit(1)
    df = pd.read_csv(csv_path, header=None, names=['image', 'label'])
    if df.empty:
        print(f"[ERROR] CSV is empty: {csv_path}")
        sys.exit(1)
    missing = [p for p in df['image'] if not os.path.exists(p)]
    if missing:
        print(f"[ERROR] {len(missing)} images listed in CSV do not exist. Example: {missing[:3]}")
        sys.exit(1)
    # Check image shapes
    for p in df['image'].sample(min(5, len(df))):
        img = cv2.imread(p)
        if img is None:
            print(f"[ERROR] Failed to read image: {p}")
            sys.exit(1)
        if img.shape[:2] != tuple(input_size):
            print(f"[WARN] Image {p} shape {img.shape[:2]} != input_size {input_size}")
    # Check labels
    if not pd.api.types.is_integer_dtype(df['label']):
        try:
            df['label'] = df['label'].astype(int)
        except Exception:
            print(f"[ERROR] Labels must be integers.")
            sys.exit(1)
    n_classes = len(df['label'].unique())
    print(f"[INFO] Found {len(df)} samples, {n_classes} classes.")
    return df, n_classes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', required=True, help='CSV with image_path,label (e.g. train.csv, no bbox columns)')
    parser.add_argument('--out', default="models/two_stage_classifier.h5", help='Output model path (.h5)')
    parser.add_argument('--input_size', type=int, nargs=2, default=[64, 64])
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--val_split', type=float, default=0.1)
    parser.add_argument('--batch', type=int, default=32)
    args = parser.parse_args()

    df, n_classes = validate_inputs(args.csv, args.input_size)
    images = []
    labels = []
    for _, row in df.iterrows():
        img = cv2.imread(row['image'])
        if img is None:
            print(f"[WARN] Could not read image: {row['image']}. Skipping.")
            continue
        images.append(cv2_resize(img, tuple(args.input_size)))
        labels.append(int(row['label']))
    images = np.array(images)
    labels = np.array(labels)

    # Remove classes with <2 samples
    from collections import Counter
    label_counts = Counter(labels)
    keep_mask = np.array([label_counts[l] >= 2 for l in labels])
    n_dropped = np.sum(~keep_mask)
    if n_dropped > 0:
        dropped_classes = sorted(set(labels[~keep_mask]))
        print(f"[WARN] Dropping {n_dropped} samples from classes with <2 samples: {dropped_classes}")
    images = images[keep_mask]
    labels = labels[keep_mask]

    X_train, X_val, y_train, y_val = train_test_split(
        images, labels, test_size=args.val_split, random_state=42, stratify=labels)
    X_train, X_val = np.array(X_train), np.array(X_val)
    y_train, y_val = np.array(y_train), np.array(y_val)

    print(f"[INFO] Training set: {X_train.shape[0]}, Validation set: {X_val.shape[0]}")
    clf = TinyClassifier(num_classes=n_classes, input_size=tuple(args.input_size))
    clf.build()
    history = clf.fit(X_train, y_train, (X_val, y_val), epochs=args.epochs)
    clf.save(args.out)
    print(f"[OK] Model saved to {args.out}")
    # Save training log
    log_path = os.path.splitext(args.out)[0] + '_log.csv'
    pd.DataFrame(history.history).to_csv(log_path, index=False)
    print(f"[OK] Training log saved to {log_path}")

if __name__ == '__main__':
    main()
