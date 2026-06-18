#!/usr/bin/env python3
"""
Small utility to rewrite Windows-style backslash paths in train_5s_spectrograms.csv
to forward slashes so they work on Linux with SpectrogramDataStream / main_spectrogram.py.

Usage:
    python fix_paths_for_linux.py
    python fix_paths_for_linux.py --in-place
    python fix_paths_for_linux.py /path/to/train_5s_spectrograms.csv --in-place
"""

import csv
import sys
from pathlib import Path


def main():
    # Parse simple args
    in_place = "--in-place" in sys.argv
    csv_arg = None
    for a in sys.argv[1:]:
        if not a.startswith("-"):
            csv_arg = a
            break

    default = Path("5sSpectrograms_tensors/train_5s_spectrograms.csv")
    csv_path = Path(csv_arg) if csv_arg else default

    if not csv_path.exists():
        print(f"ERROR: Could not find CSV at {csv_path.resolve()}")
        print("Run this script from the 'noises' directory, or pass the full path to the CSV.")
        sys.exit(1)

    if in_place:
        out_path = csv_path
    else:
        out_path = csv_path.with_name(csv_path.stem + "_linux.csv")

    print(f"Reading: {csv_path}")
    print(f"Writing: {out_path}")

    total_rows = 0
    fixed_cells = 0

    with csv_path.open("r", newline="", encoding="utf-8", errors="replace") as f_in:
        reader = csv.DictReader(f_in)
        fieldnames = reader.fieldnames or []

        # Columns that commonly contain file paths
        path_cols = [c for c in ("spectrogram_npy_path", "filename") if c in fieldnames]

        with out_path.open("w", newline="", encoding="utf-8") as f_out:
            writer = csv.DictWriter(f_out, fieldnames=fieldnames, lineterminator="\n")
            writer.writeheader()

            for row in reader:
                total_rows += 1
                for col in path_cols:
                    val = row.get(col)
                    if val and "\\" in val:
                        row[col] = val.replace("\\", "/")
                        fixed_cells += 1
                writer.writerow(row)

    print(f"Processed {total_rows:,} rows.")
    print(f"Converted {fixed_cells:,} backslash path segments to forward slashes.")
    if not in_place:
        print(f"\nNew Linux-compatible CSV created: {out_path.name}")
        print("Update your code (or copy/rename) to use this file, e.g.:")
        print(f'  csv_path="{out_path}"')
    else:
        print("File was overwritten in-place with Linux paths.")


if __name__ == "__main__":
    main()
