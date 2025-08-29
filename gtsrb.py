import os
import pandas as pd
from pathlib import Path

def generate_train_csv(train_csv, out_csv):
    df = pd.read_csv(train_csv)
    rows = []
    for _, row in df.iterrows():
        fname = row['Path']              # ← use Path column
        class_id = row['ClassId']
        xmin, ymin, xmax, ymax = row['Roi.X1'], row['Roi.Y1'], row['Roi.X2'], row['Roi.Y2']
        rows.append([fname, class_id, xmin, ymin, xmax, ymax])
    out = pd.DataFrame(rows, columns=['filename','class_id','xmin','ymin','xmax','ymax'])
    out.to_csv(out_csv, index=False)
    print(f"[OK] Wrote train CSV with {len(out)} entries → {out_csv}")

def generate_test_csv(test_csv, out_csv):
    df = pd.read_csv(test_csv)
    rows = []
    for _, row in df.iterrows():
        fname = row['Path']              # ← use Path column
        class_id = row['ClassId']
        xmin, ymin, xmax, ymax = row['Roi.X1'], row['Roi.Y1'], row['Roi.X2'], row['Roi.Y2']
        rows.append([fname, class_id, xmin, ymin, xmax, ymax])
    out = pd.DataFrame(rows, columns=['filename','class_id','xmin','ymin','xmax','ymax'])
    out.to_csv(out_csv, index=False)
    print(f"[OK] Wrote test CSV with {len(out)} entries → {out_csv}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_csv', required=True, help='Original train.csv from GTSRB')
    parser.add_argument('--test_csv', required=True, help='Original test.csv from GTSRB')
    parser.add_argument('--out_dir', required=True, help='Where to write converted CSVs')
    args = parser.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    generate_train_csv(args.train_csv, os.path.join(args.out_dir, 'train_converted.csv'))
    generate_test_csv(args.test_csv, os.path.join(args.out_dir, 'test_converted.csv'))
