
# Takes an xray PNG and saves an augmented version simulating a different machine/protocol.

import sys
import random
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
import os


def augment_xray(input_path: str, output_path: str):
    
    #grey scale
    img = Image.open(input_path).convert("L")
    arr = np.array(img, dtype=np.float32) / 255.0

    #adjust resolution
    scale = random.uniform(0.75, 1.0)
    if scale < 1.0:
        h, w = arr.shape
        small = Image.fromarray((arr * 255).astype(np.uint8)).resize((int(w * scale), int(h * scale)), Image.BILINEAR)
        arr = np.array(small.resize((w, h), Image.BILINEAR), dtype=np.float32) / 255.0

    # blur
    blur = random.uniform(0.5, 2.5)
    arr = gaussian_filter(arr, sigma=blur)

    # noise
    noise_std = random.uniform(0.04, 0.12)
    arr = arr + np.random.normal(0, noise_std, arr.shape).astype(np.float32)
    arr = np.clip(arr, 0, 1)

    # adjust contrast
    gamma = random.uniform(0.75, 1.3)
    arr = np.power(np.clip(arr, 1e-6, 1.0), gamma)

    lo = np.percentile(arr, random.uniform(1, 5))
    hi = np.percentile(arr, random.uniform(95, 99))
    arr = np.clip((arr - lo) / (hi - lo + 1e-6), 0, 1)

    Image.fromarray((arr * 255).astype(np.uint8), mode="L").save(output_path)
    print(f"Save image: {output_path}")


def augment_directory(src_root: str, dst_root: str):

    for split in ["train", "test", "val"]:
        for label in ["NORMAL", "PNEUMONIA"]:
            src_dir = os.path.join(src_root, split, label)
            dst_dir = os.path.join(dst_root, split, label)
            os.makedirs(dst_dir, exist_ok=True)

            for fname in os.listdir(src_dir):
                if not fname.lower().endswith((".png", ".jpeg", ".jpg")):
                    continue
                augment_xray(
                    os.path.join(src_dir, fname),
                    os.path.join(dst_dir, fname)
                )

if __name__ == "__main__":
    augment_directory("../chest_xray/xray_og", "../chest_xray/xray_transformed")