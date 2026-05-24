# to make a combined dataset csv using both original and transformed images
# trying to create a perfect split of original and transformed images
# also trying to ensure we don't have a transformed and original image of the SAME xray. Should ideally be different samples (because realistically we won't have both forms of xrays of same patient in same state)


import os
import csv
import pandas as pd

OG_XRAY = "../chest_xray/xray_og/original_dataset.csv"
TRANSFORMED_XRAY = "../chest_xray/xray_transformed/transform_dataset.csv"

OUTPUT_CSV = "../chest_xray/combined_dataset.csv"

record = []

og_df = pd.read_csv(OG_XRAY)
tr_df = pd.read_csv(TRANSFORMED_XRAY)

print("len og:", len(og_df))
print("len transformed:", len(tr_df))

for i in range(len(og_df)):

    if i%2==0:
        sample = og_df.iloc[i]
        record.append({
            "path": sample.path,
            "label": sample.label,
            "split": sample.split,
        })
    else:
        sample = tr_df.iloc[i]
        record.append({
            "path": sample.path,
            "label": sample.label,
            "split": sample.split,
        })

# for i in range(5):
#     print(record[i])

with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["path", "label", "split"])
    writer.writeheader()
    writer.writerows(record)
