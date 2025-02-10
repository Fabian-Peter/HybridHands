import os
import sys
from pathlib import Path
from collections import defaultdict

def main(directory, start_index):
    dir_path = Path(directory)
    # List all .jpg and .json files in the directory.
    all_files = list(dir_path.glob("*.jpg")) + list(dir_path.glob("*.json"))
    
    # Group files by their original numeric prefix.
    # We assume filenames are of the form "00000000.jpg" or "00000000.json".
    groups = defaultdict(list)
    for f in all_files:
        groups[f.stem].append(f)
    
    # Sort groups by the numeric value of the original prefix.
    sorted_keys = sorted(groups.keys(), key=lambda x: int(x))
    print(f"Found {len(sorted_keys)} groups (pairs) in {directory}")

    # First pass: rename each file to a temporary name to avoid collisions.
    for key in sorted_keys:
        for f in groups[key]:
            temp_name = "temp_" + f.name
            f.rename(f.with_name(temp_name))
    print("First pass complete: temporary names assigned.")

    # Second pass: rename each group using the same new index.
    current_index = start_index
    for key in sorted_keys:
        # Find all files in this group that now have the temporary prefix.
        group_files = list(dir_path.glob(f"temp_{key}.*"))
        new_base = f"{current_index:08d}"
        for f in group_files:
            f.rename(f.with_name(new_base + f.suffix))
        current_index += 1

    print(f"Renaming complete. Files are now numbered in pairs starting from {str(start_index).zfill(8)}.")

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python rename_dataset_pairs.py <directory> <start_index>")
        sys.exit(1)
    directory = sys.argv[1]
    start_index = int(sys.argv[2])
    main(directory, start_index)