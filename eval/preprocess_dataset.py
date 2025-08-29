"""Preprocessing pipeline for GTSRB/BelgianTS datasets.

Features:
- Generates classifier crops (resized, optionally black-bordered) and CSV for two-stage pipeline.
- Generates YOLO-format label files from GTSRB/BelgianTS annotations.

Assumes:
- Dataset root contains images and a CSV with columns: filename,class_id[,xmin,ymin,xmax,ymax]
- If bounding boxes are present, uses them; else, uses full image.
"""
import sys
import os
# Add project root to sys.path for robust relative imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import os
import csv
import shutil
from pathlib import Path
import cv2
import numpy as np

def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)

def safe_path(base, fname, flatten=False):
    """Join base + fname, optionally flattening subdirs."""
    if flatten:
        return os.path.join(base, os.path.basename(fname))
    return os.path.join(base, fname)

def make_classifier_crops(csv_path, img_dir, out_dir, crop_size=(64,64), use_bbox=True, flatten=False):
    """Generate crops for classifier and a CSV: image_path,label"""
    from tqdm import tqdm
    ensure_dir(out_dir)
    out_csv = os.path.join(out_dir, 'classifier_labels.csv')
    if not os.path.exists(csv_path):
        print(f"[ERROR] CSV file not found: {csv_path}")
        return
    if not os.path.isdir(img_dir):
        print(f"[ERROR] Image directory not found: {img_dir}")
        return
    from two_stage.detector import detect_and_crop
    DETECTOR_KWARGS = dict(min_area=200, eps_coef=0.03)  # min_area is int, eps_coef is float
    with open(csv_path, 'r') as f, open(out_csv, 'w', newline='') as fout:
        reader = list(csv.reader(f))
        writer = csv.writer(fout)
        for idx, row in enumerate(tqdm(reader, desc='Classifier crops (detector)')):
            if not row or row[0].startswith('#'):
                continue
            if len(row) < 2:
                print(f"[WARN] Row {idx} malformed: {row}")
                continue
            fname, class_id = row[0], row[1]
            img_path = os.path.join(img_dir, fname)
            if not os.path.exists(img_path):
                print(f"[WARN] Image not found: {img_path}")
                continue
            img = cv2.imread(img_path)
            if img is None:
                print(f"[WARN] Failed to read image: {img_path}")
                continue
            try:
                crops = detect_and_crop(img, **DETECTOR_KWARGS)
                if not crops:
                    print(f"[WARN] Detector found no crops in {img_path}")
                    continue
                for i, crop in enumerate(crops):
                    crop = cv2.resize(crop, crop_size, interpolation=cv2.INTER_AREA)
                    bordered = cv2.copyMakeBorder(crop, 2,2,2,2, cv2.BORDER_CONSTANT, value=(0,0,0))
                    out_img_path = safe_path(out_dir, f"{Path(fname).stem}_detcrop{i}.png", flatten=True)
                    ensure_dir(os.path.dirname(out_img_path))
                    cv2.imwrite(out_img_path, bordered)
                    writer.writerow([out_img_path, class_id])
            except Exception as e:
                print(f"[ERROR] Exception processing {img_path}: {e}")
    print(f"Classifier crops and CSV written to {out_dir}")

def make_yolo_labels(csv_path, img_dir, yolo_img_dir, yolo_label_dir, img_wh=(64,64), flatten=False):
    """Generate YOLO-format label files and copy images to yolo_img_dir."""
    from tqdm import tqdm
    ensure_dir(yolo_img_dir)
    ensure_dir(yolo_label_dir)
    if not os.path.exists(csv_path):
        print(f"[ERROR] CSV file not found: {csv_path}")
        return
    if not os.path.isdir(img_dir):
        print(f"[ERROR] Image directory not found: {img_dir}")
        return
    with open(csv_path, 'r') as f:
        reader = list(csv.reader(f))
        for idx, row in enumerate(tqdm(reader, desc='YOLO labels')):
            if not row or row[0].startswith('#'):
                continue
            if len(row) < 2:
                print(f"[WARN] Row {idx} malformed: {row}")
                continue
            fname, class_id = row[0], row[1]
            img_path = os.path.join(img_dir, fname)
            if not os.path.exists(img_path):
                print(f"[WARN] Image not found: {img_path}")
                continue
            img = cv2.imread(img_path)
            if img is None:
                print(f"[WARN] Failed to read image: {img_path}")
                continue
            try:
                h, w = img.shape[:2]
                if len(row) >= 6:
                    xmin, ymin, xmax, ymax = map(int, row[2:6])
                else:
                    xmin, ymin, xmax, ymax = 0, 0, w, h
                x_center = (xmin + xmax) / 2 / w
                y_center = (ymin + ymax) / 2 / h
                bw = (xmax - xmin) / w
                bh = (ymax - ymin) / h
                yolo_label = f"{class_id} {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}\n"

                out_img_path = safe_path(yolo_img_dir, fname, flatten=flatten)
                out_label_path = safe_path(yolo_label_dir, Path(fname).stem + '.txt', flatten=flatten)

                ensure_dir(os.path.dirname(out_img_path))
                ensure_dir(os.path.dirname(out_label_path))

                shutil.copy(img_path, out_img_path)
                with open(out_label_path, 'w') as fout:
                    fout.write(yolo_label)
            except Exception as e:
                print(f"[ERROR] Exception processing {img_path}: {e}")
    print(f"YOLO images and labels written to {yolo_img_dir}, {yolo_label_dir}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', required=True, help='CSV with filename,class_id[,xmin,ymin,xmax,ymax]')
    parser.add_argument('--img_dir', required=True, help='Directory with images')
    parser.add_argument('--out_dir', required=True, help='Output directory for classifier crops')
    parser.add_argument('--yolo_img_dir', required=True, help='Output directory for YOLO images')
    parser.add_argument('--yolo_label_dir', required=True, help='Output directory for YOLO labels')
    parser.add_argument('--crop_size', type=int, nargs=2, default=[64,64])
    parser.add_argument('--no_bbox', action='store_true', help='Ignore bbox, use full image')
    parser.add_argument('--flatten', action='store_true', help='Flatten paths (ignore subdirs)')
    args = parser.parse_args()

    make_classifier_crops(args.csv, args.img_dir, args.out_dir,
                          tuple(args.crop_size), use_bbox=not args.no_bbox, flatten=args.flatten)
    make_yolo_labels(args.csv, args.img_dir, args.yolo_img_dir,
                     args.yolo_label_dir, flatten=args.flatten)
