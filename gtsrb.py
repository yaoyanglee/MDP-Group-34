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

def convert_bbox_csv_to_simple(input_csv, output_csv):
    """Convert a bbox CSV (filename,class_id,xmin,ymin,xmax,ymax) to simple test.csv (filename,class_id) format."""
    df = pd.read_csv(input_csv)
    if not {'filename','class_id'}.issubset(df.columns):
        raise ValueError(f"Input CSV must have columns: filename,class_id,... (got {df.columns})")
    df_simple = df[['filename','class_id']]
    df_simple.to_csv(output_csv, index=False, header=False)
    print(f"[OK] Converted {input_csv} → {output_csv} (filename,class_id)")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_csv', required=True, help='Original train.csv from GTSRB')
    parser.add_argument('--test_csv', required=True, help='Original test.csv from GTSRB')
    parser.add_argument('--out_dir', required=True, help='Where to write converted CSVs')
    args = parser.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    train_converted = os.path.join(args.out_dir, 'train_converted.csv')
    test_converted = os.path.join(args.out_dir, 'test_converted.csv')
    train_simple = os.path.join(args.out_dir, 'train.csv')
    test_simple = os.path.join(args.out_dir, 'test.csv')

    generate_train_csv(args.train_csv, train_converted)
    generate_test_csv(args.test_csv, test_converted)

    # Always convert both train and test bbox CSVs to simple CSVs
    convert_bbox_csv_to_simple(train_converted, train_simple)
    convert_bbox_csv_to_simple(test_converted, test_simple)
    print(f"[OK] All four CSVs written to {args.out_dir}:")
    print(f"  - train_converted.csv (with bbox)")
    print(f"  - test_converted.csv (with bbox)")
    print(f"  - train.csv (no bbox)")
    print(f"  - test.csv (no bbox)")
