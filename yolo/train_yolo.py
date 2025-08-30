"""YOLOv8 training pipeline using ultralytics.

Assumes images and labels are in YOLO format (see preprocess_dataset.py).
Uses train_converted.csv and test_converted.csv (with bounding boxes) for training/validation.
Generates a data.yaml and launches training.
"""
import os
import yaml
from pathlib import Path

def write_data_yaml(train_img_dir, val_img_dir, nc, names, out_path):
    # Validate paths
    if not os.path.isdir(train_img_dir):
        print(f"[ERROR] Training image directory not found: {train_img_dir}")
        raise FileNotFoundError(train_img_dir)
    if not os.path.isdir(val_img_dir):
        print(f"[ERROR] Validation image directory not found: {val_img_dir}")
        raise FileNotFoundError(val_img_dir)
    data = {
        'train': str(Path(train_img_dir).absolute()),
        'val': str(Path(val_img_dir).absolute()),
        'nc': int(nc),
        'names': names
    }
    try:
        with open(out_path, 'w') as f:
            yaml.dump(data, f)
        print(f"[INFO] Wrote {out_path}")
    except Exception as e:
        print(f"[ERROR] Failed to write {out_path}: {e}")
        raise

def run_yolo_train(data_yaml, model='yolov8n.pt', epochs=20, imgsz=64, batch=32, device='cpu', project='yolo_train', name='exp'):
    import os
    if not os.path.exists(model):
        print(f"[ERROR] YOLO model weights not found: {model}")
        raise FileNotFoundError(model)
    if not os.path.exists(data_yaml):
        print(f"[ERROR] data.yaml not found: {data_yaml}")
        raise FileNotFoundError(data_yaml)
    try:
        from ultralytics import YOLO
    except Exception as e:
        print(f"[ERROR] ultralytics import failed: {e}")
        raise
    try:
        yolo = YOLO(model)
        print(f"[INFO] Starting YOLO training with model={model}, data={data_yaml}")
        yolo.train(data=data_yaml, epochs=epochs, imgsz=imgsz, batch=batch, device=device, project=project, name=name)
    except Exception as e:
        print(f"[ERROR] YOLO training failed: {e}")
        raise

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_csv', required=True, help='train_converted.csv (with bbox)')
    parser.add_argument('--val_csv', required=True, help='test_converted.csv (with bbox)')
    parser.add_argument('--img_dir', required=True, help='Directory with all images')
    parser.add_argument('--nc', type=int, required=True)
    parser.add_argument('--names', nargs='+', required=True, help='Class names')
    parser.add_argument('--model', default='yolov8n.pt')
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--imgsz', type=int, default=64)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--project', default='models')
    parser.add_argument('--name', default='yolo')
    args = parser.parse_args()

    # The YOLO training expects images and labels in YOLO format directories.
    # Assume user has already run preprocessing to generate these from train_converted.csv and test_converted.csv.
    train_img_dir = os.path.join(args.img_dir, 'train')
    val_img_dir = os.path.join(args.img_dir, 'val')
    # If you use a single directory for all images, set train_img_dir = val_img_dir = args.img_dir
    # For now, use args.img_dir for both
    train_img_dir = args.img_dir
    val_img_dir = args.img_dir
    data_yaml = 'data.yaml'
    write_data_yaml(train_img_dir, val_img_dir, args.nc, args.names, data_yaml)
    run_yolo_train(data_yaml, model=args.model, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, device=args.device, project=args.project, name=args.name)