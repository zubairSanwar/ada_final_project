# JUST TO GO THROUGH DATASET AND SAVE AS CSV

import os
import csv

#kept these as global vars so it's easier to change for different datsets (easier than typing paths in terminal imo)
DIR = "../chest_xray/xray_transformed"
OUTPUT_CSV = "../chest_xray/xray_transformed/dataset.csv"

SPLITS = ["train", "val", "test"]
CLASSIFICATION = {"NORMAL": 0, "PNEUMONIA": 1}

def record_info():
    records = []

    for split in SPLITS:
        for class_name, label in CLASSIFICATION.items():
            folder = f"{DIR}/{split}/{class_name}"
            for filename in sorted(os.listdir(folder)):
                if filename.lower().endswith("jpeg"):
                    records.append({
                        "path": f"{folder}/{filename}",
                        "label": label,
                        "split": split,
                    })

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "label", "split"])
        writer.writeheader()
        writer.writerows(records)

if __name__ == "__main__":
    record_info();
    print("done reading files")