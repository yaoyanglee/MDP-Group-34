"""
Hybrid preprocessing for GTSRB/BelgianTS.

Outputs:
1. Two-stage classifier crops + CSV
   -> train.csv / test.csv : filename,class_id
2. YOLO dataset (original images only) with normalized .txt labels
   -> one .txt per image in same folder as image
"""

import os, csv, shutil
from pathlib import Path
import cv2
from tqdm import tqdm
from two_stage.detector import detect_and_crop

def ensure_dir(p): Path(p).mkdir(parents=True, exist_ok=True)

# ------------------------
# Two-stage classifier
# ------------------------
def make_classifier_set(csv_in, img_dir, out_dir, crop_size=(64,64)):
    """Run detector, save crops, and write classifier CSV (fname,label)."""
    ensure_dir(out_dir)
    out_csv = os.path.join(out_dir, "train.csv" if "train" in csv_in.lower() else "test.csv")
    passed, skipped = 0, 0

    with open(csv_in, "r") as f, open(out_csv, "w", newline="") as fout:
        reader, writer = list(csv.reader(f)), csv.writer(fout)
        if reader and ("filename" in reader[0][0].lower() or "path" in reader[0][0].lower()):
            reader = reader[1:]

        for idx, row in enumerate(tqdm(reader, desc="Two-stage crops")):
            if len(row) < 2: continue
            fname, class_id = row[0], row[1]
            img_path = os.path.join(img_dir, fname)
            if not os.path.exists(img_path): skipped += 1; continue
            img = cv2.imread(img_path)
            if img is None: skipped += 1; continue

            crops = detect_and_crop(img, min_area=200, eps_coef=0.03)
            if not crops: skipped += 1; continue

            for ci, crop in enumerate(crops):
                crop = cv2.resize(crop, crop_size)
                crop_name = f"{Path(fname).stem}_crop{ci}.png"
                crop_path = os.path.join(out_dir, crop_name)
                cv2.imwrite(crop_path, crop)
                writer.writerow([crop_path, class_id])
                passed += 1

    print(f"[INFO] Wrote classifier CSV {out_csv}, {passed} crops, {skipped} skipped.")

# ------------------------
# YOLO labels (.txt) generation
# ------------------------
def make_yolo_txt_labels(csv_in, img_dir, out_img_dir, out_label_dir):
    """
    Generate YOLO-compatible .txt labels.
    csv_in: CSV with filename,class_id,xmin,ymin,xmax,ymax
    img_dir: folder with images
    out_img_dir: folder to copy images into
    """
    ensure_dir(out_img_dir)
    ensure_dir(out_label_dir)
    with open(csv_in, "r") as f:
        reader = list(csv.reader(f))
        if reader and ("filename" in reader[0][0].lower() or "path" in reader[0][0].lower()):
            reader = reader[1:]

        for row in tqdm(reader, desc="YOLO .txt labels"):
            if len(row) < 6:
                continue
            fname, cls, xmin, ymin, xmax, ymax = row
            img_path = os.path.join(img_dir, fname)
            if not os.path.exists(img_path):
                print(f"[WARN] Image missing: {img_path}")
                continue

                # Copy image to output dir
            out_img_path = os.path.join(out_img_dir, os.path.basename(fname))
            ensure_dir(os.path.dirname(out_img_path))
            shutil.copy(img_path, out_img_path)
            # Read image shape
            img = cv2.imread(img_path)
            if img is None:
                print(f"[WARN] Failed to read image: {img_path}")
                continue  # or skip writing bbox
            h, w = img.shape[:2]
            # Normalize bbox
            x_center = (float(xmin) + float(xmax)) / 2 / w
            y_center = (float(ymin) + float(ymax)) / 2 / h
            bw = (float(xmax) - float(xmin)) / w
            bh = (float(ymax) - float(ymin)) / h
            # Write .txt file to yolo_label_dir
            txt_file = os.path.splitext(os.path.basename(fname))[0] + ".txt"
            txt_path = os.path.join(out_label_dir, txt_file)
            with open(txt_path, "w") as ftxt:
                ftxt.write(f"{cls} {x_center} {y_center} {bw} {bh}\n")

    print(f"[INFO] YOLO .txt labels generated in {out_img_dir}")

# ------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_twostage", required=True, help="CSV for two-stage (filename,class_id)")
    parser.add_argument("--csv_yolo", required=True, help="CSV for YOLO (filename,class_id,xmin,ymin,xmax,ymax)")
    parser.add_argument("--img_dir", required=True, help="Directory with images")
    parser.add_argument("--cls_out", required=True, help="Output dir for classifier crops")
    parser.add_argument("--yolo_img_out", required=True, help="Output directory for YOLO images")
    parser.add_argument("--yolo_label_out", required=True, help="Output directory for YOLO labels")
    args = parser.parse_args()

    make_classifier_set(args.csv_twostage, args.img_dir, args.cls_out)
    make_yolo_txt_labels(args.csv_yolo, args.img_dir, args.yolo_img_out, args.yolo_label_out)
