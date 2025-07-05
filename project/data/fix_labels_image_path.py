import pandas as pd
import os
import argparse

def fix_image_paths(labels_csv, output_csv=None):
    df = pd.read_csv(labels_csv)
    def fix_path(p):
        p = str(p)
        return p if p.startswith('images/') else os.path.join('images', p)
    df['image_path'] = df['image_path'].apply(fix_path)
    if output_csv is None:
        output_csv = os.path.splitext(labels_csv)[0] + '_fixed.csv'
    df.to_csv(output_csv, index=False)
    print(f"Fixed CSV saved to {output_csv}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepend images/ to image_path in labels.csv')
    parser.add_argument('labels_csv', help='Path to labels.csv')
    parser.add_argument('--output_csv', default=None, help='Output CSV path (default: labels_fixed.csv)')
    args = parser.parse_args()
    fix_image_paths(args.labels_csv, args.output_csv) 